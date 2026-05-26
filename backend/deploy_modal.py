#!/usr/bin/env python3
"""
Modal deployment script for OpenFlipbook.

Usage:
    modal deploy backend/deploy_modal.py

This deploys the Modal app so it can be called via:
    from backend.modal_generation import generate_image
    result = generate_image.remote("a beautiful sunset over the ocean", 512, 512)
"""

import modal

# Image generation dependencies — FLUX.1-schnell is fast (4 steps)
IMAGE_DEPS = [
    "diffusers>=0.31.0",
    "transformers>=4.46.0",
    "accelerate>=1.2.0",
    "torch>=2.5.0",
    "torchvision>=0.20.0",
    "pillow>=11.0.0",
    "numpy>=1.26.0",
    "huggingface_hub>=0.26.0",
]

app = modal.App("open-flipbook")

# Attach HF_TOKEN secret for FLUX.1 gated model access
hf_secret = modal.Secret.from_name("hf-token")

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*IMAGE_DEPS)
    .env({
        "HF_HOME": "/modal_cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Faster model downloads
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
    })
)


@app.cls(
    image=image,
    secrets=[hf_secret],  # HF_TOKEN for FLUX.1 gated model
    gpu="A10",  # $0.000306/sec — ~$0.004 per 512x512 image
    timeout=300,
    scaledown_window=120,  # Keep warm for 2 min between calls
    retries=1,
)
class FluxGenerator:
    """FLUX.1-schnell generator on Modal A10 GPU."""

    @modal.enter()
    def setup(self):
        """Load model once per warm container."""
        import torch
        from diffusers import FluxPipeline

        print("[Modal] Loading FLUX.1-schnell on A10...")
        self.pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
        )
        self.pipeline = self.pipeline.to("cuda")
        self.pipeline.enable_attention_slicing()
        self.pipeline.enable_vae_slicing()
        print("[Modal] FLUX.1-schnell ready")

    @modal.method()
    def generate(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        seed: int = None,
    ) -> dict:
        """
        Generate an image. Returns dict with base64 + metadata.

        Cost: ~$0.004 per 512x512 image (A10, ~12s generation)
              ~$0.010 per 1024x1024 image (A10, ~30s generation)
        """
        import io
        import base64
        import time
        import torch
        from PIL import Image

        start = time.time()

        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        generator = torch.Generator(device="cuda").manual_seed(seed)

        result = self.pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=4,  # schnell is fast
            generator=generator,
            guidance_scale=0.0,  # No guidance needed
        )

        image = result.images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        elapsed = time.time() - start

        return {
            "image_b64": image_b64,
            "seed": seed,
            "timing": round(elapsed, 2),
            "width": width,
            "height": height,
            "gpu": "A10",
        }


@app.function(
    image=image,
    secrets=[hf_secret],
    gpu="A10",
    timeout=300,
    scaledown_window=120,
)
def generate_image(prompt: str, width: int = 512, height: int = 512, seed: int = None) -> dict:
    """
    Standalone function — creates its own pipeline per call.
    Use FluxGenerator class above for better container reuse.
    """
    import io
    import base64
    import time
    import torch
    from diffusers import FluxPipeline
    from PIL import Image

    start = time.time()

    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    )
    pipeline = pipeline.to("cuda")
    pipeline.enable_attention_slicing()
    pipeline.enable_vae_slicing()

    if seed is None:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipeline(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=4,
        generator=generator,
        guidance_scale=0.0,
    )

    image = result.images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    elapsed = time.time() - start

    return {
        "image_b64": image_b64,
        "seed": seed,
        "timing": round(elapsed, 2),
        "width": width,
        "height": height,
    }


@app.local_entrypoint()
def main():
    """Test the Modal deployment locally."""
    print("Testing Modal deployment...")
    print("NOTE: This runs locally. Use `modal deploy backend/deploy_modal.py` to deploy.")
    print()
    print("To test the deployed version:")
    print("  from backend.deploy_modal import FluxGenerator")
    print("  result = FluxGenerator().generate.remote('a sunset', 512, 512)")


if __name__ == "__main__":
    # Run locally (not on Modal)
    main()
