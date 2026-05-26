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
│  │   Search    │  │  (FLUX.1/    │  │   (LTX-Video)   │  │
│  │  (Tavily/   │  │   SDXL)      │  │                  │  │
│  │  DuckDuckGo)│  │              │  │  (transitions)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/jessiorg/open-flipbook.git
cd open-flipbook
pip install -r requirements.txt
```

### 2. Run Backend

```bash
cd backend
python main.py
```

### 3. Open Frontend

```bash
# Open in browser
open frontend/index.html
# Or serve it
python -m http.server 8080 --directory frontend
```

### 4. GPU / Model Notes

- **Image**: FLUX.1-dev via Hugging Face diffusers (or SDXL fallback)
- **Video**: LTX-Video for animated transitions
- **Agent**: Tavily API + local LLM (or DeepSeek/MiniMax via CLAUDE.md config)
- Runs on a single GPU (RTX 4090 / A100). Start at 512x512 or 720p.

## Core Flow

1. **Text prompt** → agent search → detailed image prompt → generate full-screen image
2. **Click (x,y + image)** → agent interprets → new prompt → next image **or** LTX-Video transition
3. Session state preserves conversation history + last image

## Project Structure

```
open-flipbook/
├── backend/
│   ├── main.py              # FastAPI + WebSocket server
│   ├── agent.py             # Agentic search + prompt engineering
│   ├── generator.py         # Image + video generation
│   ├── session.py           # Session state management
│   └── templates.py         # Prompt templates
├── frontend/
│   └── index.html           # Pure canvas + WS client, no frameworks
├── requirements.txt
└── README.md
```

## Performance Targets

| Mode | Resolution | Speed | Notes |
|------|-----------|-------|-------|
| Prototype | 512x512 | ~10s/image | FLUX.1-schnell |
| Production | 1024x1024 | ~30s/image | FLUX.1-dev |
| Video | 768x768 16f | ~60s | LTX-Video |

## Credits

Concept by Zain Shah, Eddie Jiao, Drew Carr — flipbook.page

OpenFlipbook is an open-source reconstruction.
