# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import subprocess
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    KDPM2AncestralDiscreteScheduler,
    AutoencoderKL
)

MODEL_URL = "https://weights.replicate.delivery/default/corcelio/mobius/model.tar"
MODEL_CACHE = "checkpoints"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            cache_dir="checkpoints",
            torch_dtype=torch.float16
        )
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "Corcelio/mobius",
            cache_dir="checkpoints",
            vae=self.vae,
            torch_dtype=torch.float16
        )
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe.to('cuda')

    def predict(
        self,
        prompt: str = Input(description="Input prompt. Consider using these key words: best quality, HD, '*aesthetic*'.", default="The Exegenesis of the soul, captured within a boundless well of starlight, pulsating and vibrating wisps, chiaroscuro, humming transformer"),
        negative_prompt: str = Input(description="Input negative prompt", default=None),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        guidance_scale: float = Input(
            description="Classifier-free guidance. 3.5 for realism, 7 for anime", ge=1, le=50, default=7
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=50
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            clip_skip=3
        ).images[0]

        output_path = "/tmp/output.jpg"
        image.save(output_path)
        return Path(output_path)
