#!/usr/bin/env python3
"""
OpenFlipbook — Modal Web Server

Deploy:
    cd ~/claudec/open-flipbook
    modal deploy backend/modal_server.py

Runs FastAPI + WebSocket on Modal A10 GPU.
Everything in one file — no path import issues.
"""

import modal
import os

# ── Dependencies ──────────────────────────────────────────────────────────────

DEPS = [
    "fastapi>=0.115.0",
    "uvicorn[standard]==0.30.0",
    "websockets==12.0",
    "python-multipart==0.0.9",
    "diffusers>=0.31.0",
    "transformers>=4.46.0",
    "accelerate>=1.2.0",
    "torch>=2.5.0",
    "torchvision>=0.20.0",
    "pillow>=11.0.0",
    "numpy>=1.26.0",
    "huggingface_hub>=0.26.0",
    "langchain==0.3.12",
    "langchain-community==0.3.12",
    "langchain-openai==0.2.10",
    "python-dotenv==1.0.0",
    "aiofiles==24.1.0",
    "httpx>=0.28.0",
]

hf_secret = modal.Secret.from_name("hf-token")
minimax_secret = modal.Secret.from_name("minimax-key")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*DEPS)
    .env({
        "HF_HOME": "/modal_cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "LLM_PROVIDER": "minimax",  # Use MiniMax for LLM calls
    })
)

app = modal.App("open-flipbook-web")


# ─────────────────────────────────────────────────────────────────────────────
# Session Management
# ─────────────────────────────────────────────────────────────────────────────

import uuid, time
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Message:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    image_b64: Optional[str] = None

@dataclass
class Session:
    id: str
    messages: list = field(default_factory=list)
    last_image_b64: Optional[str] = None
    last_image_description: Optional[str] = None
    last_prompt: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    is_generating: bool = False

    def add_user_message(self, content, image_b64=None):
        self.messages.append(Message(role="user", content=content, image_b64=image_b64))
        self.last_active = time.time()

    def add_assistant_message(self, content, image_b64=None):
        self.messages.append(Message(role="assistant", content=content, image_b64=image_b64))
        self.last_active = time.time()
        if image_b64:
            self.last_image_b64 = image_b64

    def get_history_text(self) -> str:
        history = []
        for msg in self.messages[-10:]:
            role = "User" if msg.role == "user" else "Assistant"
            history.append(f"{role}: {msg.content}")
        return "\n".join(history)

class SessionManager:
    def __init__(self, max_sessions=100):
        self.sessions = {}
        self.max_sessions = max_sessions

    def create(self):
        sid = str(uuid.uuid4())[:8]
        session = Session(id=sid)
        self.sessions[sid] = session
        self._cleanup()
        return session

    def get(self, sid):
        return self.sessions.get(sid)

    def _cleanup(self):
        if len(self.sessions) > self.max_sessions:
            oldest = sorted(self.sessions.items(), key=lambda x: x[1].last_active)
            for sid, _ in oldest[:10]:
                del self.sessions[sid]


# ─────────────────────────────────────────────────────────────────────────────
# Agent — Search + Prompt Engineering
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_QUERY_TEMPLATE = "Given the user's exploration topic: {topic}\n\nGenerate 1 focused search query. Be specific and factual.\n\nReturn only the query string."

GROUNDING_TEMPLATE = "TOPIC: {topic}\n\nSEARCH RESULTS:\n{search_results}\n\nExtract 3-5 key factual details for image generation. Format as bullet points."

CLICK_INTERPRET_TEMPLATE = "PREVIOUS IMAGE: {image_description}\n\nCLICK: x={x}, y={y} (normalized 0-1)\n\nWhat is the user trying to explore? Respond:\nlabel: <short label>\nexplore: <detailed prompt>"

IMAGE_PROMPT_TEMPLATE = """You are generating a full-screen illustrative image for Flipbook — an infinite visual browser.

CONTEXT: {context}
USER ACTION: {user_action}

Create a visually rich, detailed image that:
1. Explores the topic in depth with accurate, grounded information
2. Uses editorial/illustrative style — cinematic lighting, visual hierarchy
3. ALL text rendered as painted pixels within the scene (no text overlays)
4. Fills the viewport, main subject large and central
5. Includes depth cues, compositional balance

STYLE: Rich detail, educational, cinematic. No UI chrome.

WINDOW: {width}x{height} pixels

GROUNDING (from web search):
{grounding}

Generate a detailed image prompt."""


class Agent:
    def __init__(self):
        self.llm_provider = os.getenv("LLM_PROVIDER", "minimax")
        self._init_llm()

    def _init_llm(self):
        if self.llm_provider == "minimax":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model="MiniMax-M2.7",
                api_key=os.getenv("MINIMAX_API_KEY", ""),
                base_url="https://api.minimax.io/v1",
            )
        elif self.llm_provider == "deepseek":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model="deepseek-chat",
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                base_url="https://api.deepseek.com/anthropic",
            )
        else:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model="gpt-4o-mini")

    async def search(self, query):
        try:
            from langchain_community.tools import DuckDuckGoGoSearchResults
            tool = DuckDuckGoGoSearchResults(max_results=5)
            results = await tool.ainvoke({"query": query})
            parsed = []
            if isinstance(results, str):
                for line in results.split("\n"):
                    if ": " in line:
                        parts = line.split(": ", 1)
                        if len(parts) == 2:
                            parsed.append({"title": parts[0], "content": parts[1]})
            return parsed[:5]
        except Exception as e:
            print(f"[Agent] Search error: {e}")
            return []

    async def generate_search_query(self, topic, history=""):
        prompt = SEARCH_QUERY_TEMPLATE.format(topic=topic)
        resp = await self.llm.ainvoke(prompt)
        return resp.content.strip()

    async def ground_prompt(self, topic, search_results):
        if not search_results:
            return "No web search results. Use your world knowledge."
        results_text = "\n".join([
            f"- {r.get('title', 'Untitled')}: {r.get('content', '')[:300]}"
            for r in search_results
        ])
        prompt = GROUNDING_TEMPLATE.format(topic=topic, search_results=results_text)
        resp = await self.llm.ainvoke(prompt)
        return resp.content.strip()

    async def interpret_click(self, image_description, click_x, click_y):
        prompt = CLICK_INTERPRET_TEMPLATE.format(
            image_description=image_description, x=click_x, y=click_y,
        )
        resp = await self.llm.ainvoke(prompt)
        label, explore = "", ""
        for line in resp.content.strip().split("\n"):
            if line.startswith("label:"):
                label = line.replace("label:", "").strip()
            elif line.startswith("explore:"):
                explore = line.replace("explore:", "").strip()
        return label, explore

    async def build_image_prompt(self, user_action, context, grounding, width, height):
        prompt = IMAGE_PROMPT_TEMPLATE.format(
            context=context,
            user_action=user_action,
            grounding=grounding,
            width=width,
            height=height,
        )
        resp = await self.llm.ainvoke(prompt)
        return resp.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Modal Web Server (ASGI)
# ─────────────────────────────────────────────────────────────────────────────

session_manager = SessionManager()
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent

# Generator HTTP endpoint (separate Modal GPU app)
GENERATOR_URL = "https://jessiorg--open-flipbook-generator-api-server.modal.run"

async def generate_image(prompt: str, width: int = 512, height: int = 512, seed: int = None) -> dict:
    """Call the generator HTTP API running on Modal GPU."""
    import httpx
    payload = {"prompt": prompt, "width": width, "height": height}
    if seed is not None:
        payload["seed"] = seed
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(f"{GENERATOR_URL}/generate", json=payload)
        resp.raise_for_status()
        return resp.json()


@app.function(
    image=image,
    secrets=[hf_secret, minimax_secret],
    timeout=300,
    scaledown_window=120,
)
@modal.asgi_app()
def web_server():
    import asyncio, json
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware

    api = FastAPI(title="OpenFlipbook")
    api.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    async def send_json(ws, data):
        await ws.send_text(json.dumps(data))

    async def send_progress(ws, progress, status):
        await send_json(ws, {"type": "progress", "progress": progress, "status": status})

    async def generate_for_user(ws, session, user_input, click_x=None, click_y=None, width=512, height=512):
        session.is_generating = True
        try:
            if click_x is not None and click_y is not None:
                label, exploration = await get_agent().interpret_click(
                    session.last_image_description or "", click_x, click_y,
                )
                user_action = f"Clicked '{label}': {exploration}"
            else:
                user_action = user_input
                label = user_input[:50]

            await send_progress(ws, 0.1, f"Exploring: {label}")

            sq = await get_agent().generate_search_query(user_action, session.get_history_text())
            await send_progress(ws, 0.2, f"Searching...")
            sr = await get_agent().search(sq)

            grounding = await get_agent().ground_prompt(user_action, sr)
            await send_progress(ws, 0.3, "Building prompt...")

            final_prompt = await get_agent().build_image_prompt(
                user_action=user_action,
                context=session.get_history_text() or "Starting fresh exploration",
                grounding=grounding,
                width=width,
                height=height,
            )

            await send_progress(ws, 0.4, "Generating image...")

            result = await generate_image(final_prompt, width, height)
            image_b64 = result["image_b64"]
            seed = result["seed"]
            gen_time = result["timing"]

            await send_progress(ws, 0.95, "Finalizing...")

            await send_json(ws, {
                "type": "image",
                "data": image_b64,
                "metadata": {
                    "label": label,
                    "prompt": final_prompt,
                    "seed": seed,
                    "search_query": sq,
                    "grounding": grounding,
                    "gen_time": gen_time,
                }
            })

            session.last_image_b64 = image_b64
            session.last_image_description = label
            session.add_user_message(user_input)
            session.add_assistant_message(label, image_b64)
            await send_progress(ws, 1.0, "Done")

        except Exception as e:
            import traceback
            traceback.print_exc()
            await send_json(ws, {"type": "error", "message": str(e)})
        finally:
            session.is_generating = False

    @api.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        session = session_manager.create()
        await send_json(ws, {"type": "session", "id": session.id})
        w, h = 512, 512

        try:
            while True:
                msg = await ws.receive_json()
                t = msg.get("type")

                if t == "start":
                    asyncio.create_task(generate_for_user(
                        ws, session, msg.get("prompt", ""),
                        width=msg.get("width", w), height=msg.get("height", h),
                    ))
                elif t == "click":
                    asyncio.create_task(generate_for_user(
                        ws, session, "",
                        click_x=msg.get("x"), click_y=msg.get("y"),
                        width=msg.get("width", w), height=msg.get("height", h),
                    ))
                elif t == "resize":
                    w, h = msg.get("width", w), msg.get("height", h)
                    await send_json(ws, {"type": "ack", "action": "resize"})
                elif t == "status":
                    await send_json(ws, {
                        "type": "status", "session_id": session.id,
                        "is_generating": session.is_generating,
                    })

        except WebSocketDisconnect:
            pass

    @api.get("/health")
    async def health():
        return {"status": "ok", "gpu": "A10"}

    @api.get("/")
    async def root():
        # Read frontend HTML from deployment directory
        import pathlib
        # Modal mounts project at /app
        html_path = pathlib.Path("/app/frontend/index.html")
        if html_path.exists():
            from fastapi.responses import HTMLResponse
            return HTMLResponse(content=html_path.read_text())
        # Fallback JSON info
        return {
            "name": "OpenFlipbook",
            "version": "0.1.0",
            "gpu": "A10",
            "model": "FLUX.1-schnell",
            "ws_endpoint": "wss://jessiorg--open-flipbook-web-web-server.modal.run/ws",
        }

    return api
