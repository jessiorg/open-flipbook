"""
OpenFlipbook — Image Generation Server (Modal GPU)

Deploy separately:
    modal deploy backend/generator_app.py

Then call via HTTP:
    http://open-flipbook-generator-<user>.modal.run/generate
"""

import modal
import os

DEPS = [
    "fastapi>=0.115.0",
    "uvicorn[standard]==0.30.0",
    "pydantic>=2.0.0",
    "diffusers>=0.31.0",
    "transformers>=4.46.0",
    "accelerate>=1.2.0",
    "torch>=2.5.0",
    "torchvision>=0.20.0",
    "pillow>=11.0.0",
    "numpy>=1.26.0",
    "huggingface_hub>=0.26.0",
]

hf_secret = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*DEPS)
    .env({
        "HF_HOME": "/modal_cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

app = modal.App("open-flipbook-generator")


@app.cls(
    image=image,
    secrets=[hf_secret],
    gpu="A10",
    timeout=300,
    scaledown_window=3600,  # Keep GPU container warm for 1 hour
)
class ImageGen:
    """SDXL Turbo on Modal A10 GPU."""

    @modal.enter()
    def setup(self):
        import torch
        from diffusers import StableDiffusionXLPipeline
        hf_token = os.getenv("HF_TOKEN", "")
        print("[Generator] Loading SDXL Turbo...")
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            token=hf_token,
            torch_dtype=torch.float16,
        )
        self.pipeline = self.pipeline.to("cuda")
        self.pipeline.enable_attention_slicing()
        self.pipeline.enable_vae_slicing()
        print("[Generator] SDXL Turbo ready")

    @modal.method()
    def generate(self, prompt: str, width: int = 512, height: int = 512, seed: int = None) -> dict:
        import io, base64, time, torch
        from PIL import Image
        start = time.time()
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = self.pipeline(
            prompt=prompt, width=width, height=height,
            num_inference_steps=4, generator=generator, guidance_scale=0.0,
        )
        image = result.images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {
            "image_b64": image_b64,
            "seed": seed,
            "timing": round(time.time() - start, 2),
            "width": width,
            "height": height,
        }


@app.function(
    image=image,
    secrets=[hf_secret],
    timeout=300,
)
@modal.asgi_app()
def api_server():
    """HTTP API server for image generation."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    api = FastAPI(title="OpenFlipbook Generator API")

    class GenerateRequest(BaseModel):
        prompt: str
        width: int = 512
        height: int = 512
        seed: int = None

    @api.get("/health")
    async def health():
        return {"status": "ok", "gpu": "A10", "model": "sdxl-turbo"}

    @api.get("/")
    async def root():
        return {"name": "OpenFlipbook Generator", "model": "sdxl-turbo", "endpoint": "/generate"}

    @api.post("/generate")
    async def generate(req: GenerateRequest):
        gen = ImageGen()
        result = gen.generate.remote(req.prompt, req.width, req.height, req.seed)
        return result

    return api
