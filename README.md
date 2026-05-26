# OpenFlipbook

**Flipbook-style generative UI** — an infinite visual browser where every pixel is AI-generated.

Every "page" is a full-screen image. Click anything and get a new image exploring that topic in depth. No HTML, no CSS, no DOM — just pixels rendered by image/video models.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Vanilla HTML/JS                         │
│              Canvas + WebSocket Client                       │
└─────────────────┬───────────────────────────────────────────┘
                  │ WebSocket / base64 images
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  Python + FastAPI                           │
│                  WebSocket Server                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Agentic   │  │   Image Gen  │  │    Video Gen     │  │
│  │   Search    │  │   (Modal     │  │   (LTX-Video)   │  │
│  │  (Tavily/   │  │   A10 GPU)   │  │                  │  │
│  │  DuckDuckGo)│  │              │  │  (transitions)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼ GPU (Modal)
            ┌───────────────┐
            │ FLUX.1-schnell│
            │  A10 GPU      │
            │  $0.004/img   │
            └───────────────┘
```

## GPU Inference via Modal

Image generation runs on **Modal** A10 GPU (~$0.004 per 512×512 image).

### Cost Math

| GPU | $/sec | 512×512 (~12s) | 1024×1024 (~30s) |
|-----|-------|----------------|------------------|
| A10 | $0.000306 | **$0.004** | $0.009 |
| L40S | $0.000542 | $0.007 | $0.016 |
| A100-40G | $0.000583 | $0.007 | $0.017 |

**$30 credits ≈ 7,500 images at 512×512**

## Setup

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/jessiorg/open-flipbook.git
cd open-flipbook
pip install -r requirements.txt
```

### 2. Install Modal & Authenticate

```bash
pip install modal
modal setup
# Opens browser — authenticate with your Modal workspace
```

### 3. Deploy Image Generation to Modal

```bash
modal deploy backend/deploy_modal.py
```

This deploys `FluxGenerator` to Modal A10 GPU. The backend server will call it remotely.

### 4. Set Environment Variables

```bash
# Agentic search (Tavily — free tier available)
export TAVILY_API_KEY=tvly-xxxx

# HuggingFace token (required for FLUX.1 gated model)
export HF_TOKEN=hf_xxxx
```

### 5. Run Backend

```bash
cd backend
python main.py
```

### 6. Open Frontend

```bash
# Any of:
open frontend/index.html
python -m http.server 8080 --directory frontend
```

## Project Structure

```
open-flipbook/
├── backend/
│   ├── main.py              # FastAPI + WebSocket server
│   ├── agent.py             # Agentic search + prompt engineering
│   ├── generator.py         # Routes to Modal GPU or local fallback
│   ├── deploy_modal.py      # Modal app deployment (A10 GPU)
│   ├── session.py           # Session state + conversation history
│   └── templates.py         # Flipbook-style prompt templates
├── frontend/
│   └── index.html           # Pure canvas + WebSocket client
├── requirements.txt
└── README.md
```

## Performance

| Mode | Resolution | Speed | Modal Cost |
|------|-----------|-------|-----------|
| Prototype | 512×512 | ~12s | **$0.004** |
| Full | 1024×1024 | ~30s | $0.009 |
| Video | 768×768 16f | ~60s | $0.018 (future) |

## Key Features

- **Pure canvas UI** — zero HTML/CSS/JS frameworks, only `<canvas>` + WebSocket
- **Click-to-explore** — click anywhere on an image to go deeper
- **Agentic grounding** — web search + LLM for factual content
- **Session continuity** — conversation history for contextual exploration
- **Modal GPU** — pay-per-second, no idle costs, auto-scaling

## API Keys Needed

| Service | Purpose | Free Tier |
|---------|---------|-----------|
| HuggingFace | FLUX.1 model access | No (gated) |
| Tavily | Web search grounding | Yes (1000q/month) |
| OpenAI/DeepSeek/MiniMax | LLM prompt engineering | Yes |

## Credits

Concept by Zain Shah, Eddie Jiao, Drew Carr — [flipbook.page](https://flipbook.page)

OpenFlipbook is an independent open-source reconstruction.
