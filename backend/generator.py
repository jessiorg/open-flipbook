"""
Image and video generation for OpenFlipbook.
Routes to Modal GPU when available, falls back to local generation.
"""

import os
import io
import base64
import asyncio
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from typing import Optional


# Modal routing — set MODAL=1 to force local, unset to use Modal
USE_MODAL = os.getenv("MODAL", "") == ""


class Generator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_pipeline = None
        self.video_pipeline = None
        self._modal_generator = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def modal_generator(self):
        """Lazy-load Modal generator to avoid import overhead."""
        if self._modal_generator is None:
            # Use the deployed Modal class — call .generate.remote() for image gen
            # Requires: modal deploy backend/deploy_modal.py
            from .deploy_modal import FluxGenerator
            self._modal_generator = FluxGenerator()
        return self._modal_generator

    def load_image_model(self, model_id: str = "black-forest-labs/FLUX.1-schnell"):
        """Load image generation pipeline locally (fallback when Modal unavailable)."""
        if self.image_pipeline is not None:
            return

        print(f"[Generator] Loading image model: {model_id} on {self.device}")

        try:
            from diffusers import FluxPipeline

            self.image_pipeline = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            )
            self.image_pipeline = self.image_pipeline.to(self.device)
            self.image_pipeline.enable_attention_slicing()
            self.image_pipeline.enable_vae_slicing()
            print("[Generator] Image model loaded successfully")

        except Exception as e:
            print(f"[Generator] Failed to load {model_id}: {e}")
            self._load_sdxl()

    def _load_sdxl(self):
        """Fallback to Stable Diffusion XL."""
        try:
            from diffusers import StableDiffusionXLPipeline
            self.image_pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.image_pipeline = self.image_pipeline.to(self.device)
            print("[Generator] SDXL loaded as fallback")
        except Exception as e:
            print(f"[Generator] SDXL fallback failed: {e}")
            raise RuntimeError("No image generation model available")

    async def generate_image(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        steps: int = 4,
        seed: Optional[int] = None,
    ) -> tuple[str, int, float]:
        """
        Generate an image from a text prompt.
        Routes to Modal GPU if available, otherwise uses local pipeline.

        Returns:
            tuple: (base64_image, seed_used, elapsed_seconds)
        """
        if USE_MODAL:
            return await self._generate_via_modal(prompt, width, height, seed)
        else:
            return await self._generate_local(prompt, width, height, steps, seed)

    async def _generate_via_modal(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: Optional[int],
    ) -> tuple[str, int, float]:
        """Call Modal GPU function for image generation."""
        import time
        start = time.time()

        loop = asyncio.get_event_loop()

        # Modal .remote() is blocking — run in thread to avoid blocking event loop
        result = await loop.run_in_executor(
            None,
            lambda: self.modal_generator.generate.remote(prompt, width, height, seed)
        )

        elapsed = time.time() - start
        print(f"[Generator/Modal] Image generated in {elapsed:.1f}s")

        return result["image_b64"], result["seed"], elapsed

    async def _generate_local(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: Optional[int],
    ) -> tuple[str, int, float]:
        """Generate image using local GPU (fallback)."""
        import time
        start = time.time()

        if self.image_pipeline is None:
            self.load_image_model()

        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        generator = torch.Generator(device=self.device).manual_seed(seed)

        num_steps = min(steps, 4)  # FLUX.1-schnell is fast

        print(f"[Generator/Local] Generating: {width}x{height}, steps={num_steps}, seed={seed}")

        result = self.image_pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            generator=generator,
            guidance_scale=0.0,  # schnell is guidance-free
        )

        image = result.images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        elapsed = time.time() - start
        print(f"[Generator/Local] Image generated in {elapsed:.1f}s")

        return image_b64, seed, elapsed

    async def generate_video(
        self,
        prompt: str,
        from_image: Optional[Image.Image] = None,
        num_frames: int = 16,
        width: int = 768,
        height: int = 768,
    ) -> list[str]:
        """
        Generate video frames (LTX-Video) — future work.
        Returns empty list for now.
        """
        print("[Generator] Video generation not yet implemented")
        return []


# Global generator instance
generator = Generator()
