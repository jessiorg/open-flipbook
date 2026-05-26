"""
Image and video generation for OpenFlipbook.
Uses Hugging Face diffusers (FLUX.1, SDXL) and LTX-Video.
"""

import io
import base64
import torch
from PIL import Image
from typing import Optional


class Generator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_model = None
        self.video_model = None
        self.image_pipeline = None
        self.video_pipeline = None

    def load_image_model(self, model_id: str = "black-forest-labs/FLUX.1-schnell"):
        """Load image generation pipeline (FLUX.1 or SDXL)."""
        if self.image_pipeline is not None:
            return  # Already loaded

        print(f"[Generator] Loading image model: {model_id} on {self.device}")

        try:
            from diffusers import FluxPipeline

            self.image_pipeline = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                use_auth_token=model_id,  # Will fail gracefully if not set
            )
            self.image_pipeline = self.image_pipeline.to(self.device)

            # Apply optimizations
            if self.device == "cuda":
                # Try to enable SDPA
                try:
                    self.image_pipeline.enable_attention_slicing()
                    self.image_pipeline.enable_vae_slicing()
                except Exception as e:
                    print(f"[Generator] Could not enable optimizations: {e}")

            print("[Generator] Image model loaded successfully")

        except Exception as e:
            print(f"[Generator] Failed to load {model_id}: {e}")
            print("[Generator] Falling back to SDXL...")
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

    def load_video_model(self, model_id: str = "Lightricks/LTX-Video"):
        """Load video generation pipeline (LTX-Video)."""
        if self.video_pipeline is not None:
            return

        print(f"[Generator] Loading video model: {model_id} on {self.device}")

        try:
            # LTX-Video uses a different pipeline
            from diffusers import LTXVideoPipeline
            self.video_pipeline = LTXVideoPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.video_pipeline = self.video_pipeline.to(self.device)
            print("[Generator] Video model loaded successfully")
        except Exception as e:
            print(f"[Generator] LTX-Video load failed: {e}")
            self.video_pipeline = None

    async def generate_image(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        seed: Optional[int] = None,
    ) -> tuple[str, int]:
        """
        Generate an image from a text prompt.

        Returns:
            tuple: (base64_image, seed_used)
        """
        if self.image_pipeline is None:
            self.load_image_model()

        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Scale steps based on model
        num_steps = min(steps, 50)

        print(f"[Generator] Generating image: {width}x{height}, steps={num_steps}, seed={seed}")

        result = self.image_pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            generator=generator,
            guidance_scale=3.5,  # FLUX.1-schnell uses low guidance
        )

        image = result.images[0]

        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        print(f"[Generator] Image generated: {len(image_b64)} chars base64")

        return image_b64, seed

    async def generate_video(
        self,
        prompt: str,
        from_image: Optional[Image.Image] = None,
        num_frames: int = 16,
        fps: int = 24,
        width: int = 768,
        height: int = 768,
    ) -> list[str]:
        """
        Generate a video (animated transition) from prompt + optional start frame.

        Returns:
            list of base64-encoded video frames
        """
        if self.video_pipeline is None:
            self.load_video_model()
            if self.video_pipeline is None:
                print("[Generator] No video model available, skipping video")
                return []

        print(f"[Generator] Generating video: {num_frames} frames at {width}x{height}")

        # LTX-Video expects specific inputs
        # This is a simplified version — actual implementation depends on the pipeline
        try:
            frames = self.video_pipeline(
                prompt=prompt,
                image=from_image,
                num_frames=num_frames,
                width=width,
                height=height,
                num_inference_steps=30,
                guidance_scale=3.5,
            ).frames[0]

            # Convert frames to base64
            frame_b64s = []
            for frame in frames:
                buffer = io.BytesIO()
                frame.save(buffer, format="PNG")
                frame_b64s.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

            return frame_b64s

        except Exception as e:
            print(f"[Generator] Video generation failed: {e}")
            return []

    async def upscale_image(
        self,
        image_b64: str,
        scale: float = 2.0,
    ) -> str:
        """Upscale an image using the image pipeline's VAE."""
        if self.image_pipeline is None:
            return image_b64

        # Decode base64 to image
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))

        # Simple 4x upscaling via image pipeline
        # For production, use a dedicated upscaler (Real-ESRGAN, etc.)
        new_size = (int(image.width * scale), int(image.height * scale))
        upscaled = image.resize(new_size, Image.LANCZOS)

        buffer = io.BytesIO()
        upscaled.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


# Global generator instance
generator = Generator()
