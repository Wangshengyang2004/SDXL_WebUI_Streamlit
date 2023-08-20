import os
import datetime
import io
from PIL import Image

import torch
from diffusers import (DiffusionPipeline,
                       StableDiffusionXLImg2ImgPipeline,
                       StableDiffusionXLInpaintPipeline)
from diffusers.utils import load_image


class SDXL_Pipeline:
    def __init__(self) -> None:
        self.base_model = None
        self.refiner_model = None

    def load_base_model(self, cpu_offload: bool, use_re_compile: bool = False):
        model = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

        if cpu_offload:
            model.enable_model_cpu_offload()
            offload_status = "Enabled"
        else:
            model.to("cuda")
            offload_status = "Disabled"

        compile_status = "Disabled"
        if os.name == 'posix' and use_re_compile:
            model.unet = torch.compile(model.unet, mode="reduce-overhead", fullgraph=True)
            compile_status = "Enabled"

        self.base_model = model

    def load_image_to_image_model(self):
        model = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        model.to("cuda")
        self.base_model = model

    def load_inpainting_model(self):
        model = StableDiffusionXLInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        model.to("cuda")
        self.base_model = model

    def load_refiner_model(self, cpu_offload: bool):
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base_model.text_encoder_2,
            vae=self.base_model.vae,
            torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

        if cpu_offload:
            refiner.enable_model_cpu_offload()
            offload_status = "Enabled"
        else:
            refiner.to("cuda")
            offload_status = "Disabled"

        compile_status = "Disabled"
        if os.name == 'posix':
            refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
            compile_status = "Enabled"

        self.refiner_model = refiner

    def check_prompt_length(self, prompt: str):
        if len(prompt.split()) > 77:
            raise ValueError("Prompt must be less than 77 words.")

    def base_generate(self, prompt1: str, prompt2: str, negative_prompt: str, h: int, w: int, 
                      num_steps: int, high_noise_frac: float, guidance_scale: float) -> Image:
        image = self.base_model(
            prompt=prompt1,
            prompt2=prompt2,
            negative_prompt=negative_prompt,
            height=h, width=w,
            num_inference_steps=num_steps,
            denoising_end=high_noise_frac,
            guidance_scale=guidance_scale,
            output_type="latent").images[0]
        self.save_image(image)
        return image

    def refine(self, image: Image, prompt: str, negative_prompt: str, h: int, w: int, 
               num_steps: int, high_noise_frac: float, guidance_scale: float) -> Image:
        refined_image = self.refiner_model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=h, width=w,
            num_inference_steps=num_steps,
            denoising_start=high_noise_frac,
            guidance_scale=guidance_scale,
            image=image).images[0]
        return refined_image

    def generate_and_refine(self, prompt1: str, prompt2: str, negative_prompt: str, h: int, w: int, 
                        num_steps: int, high_noise_frac: float, guidance_scale: float) -> Image:
        base_image = self.base_generate(prompt1, prompt2, negative_prompt, h, w, num_steps, high_noise_frac, guidance_scale)
        refined_image = self.refine(base_image, prompt1, negative_prompt, h, w, num_steps, high_noise_frac, guidance_scale)
        self.save_image(refined_image)
        return refined_image
    
    def save_image(self, image, save_folder=None, image_name=None):
        save_folder = save_folder or "saved_images"
        os.makedirs(save_folder, exist_ok=True)

        image_name = image_name or f"image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image_path = os.path.join(save_folder, image_name)
        image.save(image_path, "PNG")