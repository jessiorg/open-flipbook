"""
OpenFlipbook Backend — FastAPI + WebSocket server.

Handles:
- WebSocket connections for real-time image streaming
- Session state management
- Agentic search + image generation pipeline
"""

import asyncio
import json
import base64
import uuid
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from .session import SessionManager, Session
from .agent import Agent
from .generator import generator


# Global state
session_manager = SessionManager()
agent = Agent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown events."""
    print("[Server] Starting OpenFlipbook backend...")
    # Pre-warm models (optional, can lazy-load on first request)
    # generator.load_image_model()
    yield
    print("[Server] Shutting down...")


app = FastAPI(title="OpenFlipbook", version="0.1.0", lifespan=lifespan)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Protocol
# ─────────────────────────────────────────────────────────────────────────────

async def send_json(ws: WebSocket, data: dict):
    """Send JSON message to WebSocket."""
    await ws.send_text(json.dumps(data))


async def send_image(ws: WebSocket, image_b64: str, metadata: dict = None):
    """Send base64 image to WebSocket."""
    msg = {
        "type": "image",
        "data": image_b64,
        "metadata": metadata or {},
    }
    await send_json(ws, msg)


async def send_progress(ws: WebSocket, progress: float, status: str):
    """Send generation progress."""
    msg = {
        "type": "progress",
        "progress": progress,
        "status": status,
    }
    await send_json(ws, msg)


# ─────────────────────────────────────────────────────────────────────────────
# Generation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def generate_for_user(
    ws: WebSocket,
    session: Session,
    user_input: str,
    click_x: Optional[float] = None,
    click_y: Optional[float] = None,
    mode: str = "image",
    width: int = 1024,
    height: int = 1024,
):
    """
    Full pipeline:
    1. Agent builds search query
    2. Web search grounding
    3. Prompt engineering
    4. Image generation
    5. Stream result
    """
    session.is_generating = True
    session.generation_mode = mode

    try:
        # ── Step 1: Interpret click or use text input ──
        if click_x is not None and click_y is not None:
            # Vision-based click interpretation
            label, exploration_prompt = await agent.interpret_click(
                image_description=session.last_image_description or "",
                click_x=click_x,
                click_y=click_y,
            )
            user_action = f"User clicked on '{label}': {exploration_prompt}"
        else:
            user_action = user_input
            label = user_input[:50]

        await send_progress(ws, 0.1, f"Exploring: {label}")

        # ── Step 2: Generate search query + search ──
        search_query = await agent.generate_search_query(
            topic=user_action,
            history=session.get_history_text(),
        )

        await send_progress(ws, 0.2, f"Searching: {search_query[:50]}...")
        search_results = await agent.search(search_query)

        # ── Step 3: Ground the prompt ──
        grounding = await agent.ground_prompt(topic=user_action, search_results=search_results)
        await send_progress(ws, 0.3, "Building image prompt...")

        # ── Step 4: Build final image prompt ──
        context = session.get_history_text() or "Starting fresh exploration"
        final_prompt = await agent.build_image_prompt(
            user_action=user_action,
            context=context,
            grounding=grounding,
            width=width,
            height=height,
        )

        await send_progress(ws, 0.4, "Generating image...")

        # ── Step 5: Generate image ──
        if mode == "video":
            # Generate video transition first
            await send_progress(ws, 0.5, "Generating video transition...")
            # Video generation is optional / experimental
            # frames = await generator.generate_video(
            #     prompt=final_prompt,
            #     from_image=None,
            # )
            # for i, frame_b64 in enumerate(frames):
            #     await send_image(ws, frame_b64, {"frame": i, "total": len(frames)})
            #     await send_progress(ws, 0.5 + 0.4 * (i / len(frames)), f"Video frame {i+1}/{len(frames)}")

        # Always generate an image at the end
        image_b64, seed, gen_time = await generator.generate_image(
            prompt=final_prompt,
            width=width,
            height=height,
        )

        await send_progress(ws, 0.95, "Finalizing...")

        # ── Step 6: Send final image ──
        await send_image(ws, image_b64, {
            "label": label,
            "prompt": final_prompt,
            "seed": seed,
            "search_query": search_query,
            "grounding": grounding,
            "gen_time": round(gen_time, 1),
        })

        # ── Update session ──
        session.last_image_b64 = image_b64
        session.last_prompt = final_prompt
        session.last_image_description = label
        session.add_user_message(user_input)
        session.add_assistant_message(label, image_b64)

        await send_progress(ws, 1.0, "Done")

    except Exception as e:
        print(f"[Server] Generation error: {e}")
        await send_json(ws, {"type": "error", "message": str(e)})

    finally:
        session.is_generating = False


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket protocol:

    Client → Server:
      - {"type": "start", "prompt": "topic to explore"}
      - {"type": "click", "x": 0.5, "y": 0.3}
      - {"type": "resize", "width": 1024, "height": 1024}
      - {"type": "video_mode", "enabled": true}

    Server → Client:
      - {"type": "image", "data": "<base64>", "metadata": {...}}
      - {"type": "progress", "progress": 0.5, "status": "..."}
      - {"type": "error", "message": "..."}
      - {"type": "session", "id": "abc123"}
    """
    await ws.accept()
    session = session_manager.create()
    await send_json(ws, {"type": "session", "id": session.id})

    width, height = 1024, 1024
    video_mode = False

    try:
        while True:
            msg = await ws.receive_json()
            msg_type = msg.get("type")

            if msg_type == "start":
                prompt = msg.get("prompt", "")
                width = msg.get("width", width)
                height = msg.get("height", height)
                video_mode = msg.get("video_mode", video_mode)

                if not prompt:
                    await send_json(ws, {"type": "error", "message": "No prompt provided"})
                    continue

                asyncio.create_task(generate_for_user(
                    ws=ws,
                    session=session,
                    user_input=prompt,
                    click_x=None,
                    click_y=None,
                    mode="video" if video_mode else "image",
                    width=width,
                    height=height,
                ))

            elif msg_type == "click":
                x = msg.get("x", 0.5)
                y = msg.get("y", 0.5)

                if session.is_generating:
                    await send_json(ws, {"type": "error", "message": "Already generating, please wait"})
                    continue

                asyncio.create_task(generate_for_user(
                    ws=ws,
                    session=session,
                    user_input="",  # No text, just click
                    click_x=x,
                    click_y=y,
                    mode="video" if video_mode else "image",
                    width=width,
                    height=height,
                ))

            elif msg_type == "resize":
                width = msg.get("width", width)
                height = msg.get("height", height)
                await send_json(ws, {"type": "ack", "action": "resize", "width": width, "height": height})

            elif msg_type == "video_mode":
                video_mode = msg.get("enabled", False)
                await send_json(ws, {"type": "ack", "action": "video_mode", "enabled": video_mode})

            elif msg_type == "status":
                await send_json(ws, {
                    "type": "status",
                    "session_id": session.id,
                    "is_generating": session.is_generating,
                    "message_count": len(session.messages),
                })

    except WebSocketDisconnect:
        print(f"[Server] Client disconnected: {session.id}")
    except Exception as e:
        print(f"[Server] WebSocket error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Endpoints (health, info)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "gpu_available": generator.device == "cuda"}


@app.get("/")
async def root():
    return {
        "name": "OpenFlipbook",
        "version": "0.1.0",
        "endpoints": {
            "websocket": "/ws",
            "health": "/health",
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
