import streamlit as st
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
import torch
import io
from PIL import Image
import os
import datetime

@st.cache_resource()
def load_text_to_image_model(cpu_offload):
    # Load the base model for text-to-image
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    cpu_offload_enabled = False
    if cpu_offload:
        base.enable_model_cpu_offload()
        cpu_offload_enabled = True
    else:
        base.to("cuda")

    torch_compile_enabled = False
    if os.name == 'posix':
        base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
        torch_compile_enabled = True


    st.info(f'Base Model loading completed. CPU Offload: {"Enabled" if cpu_offload_enabled else "Disabled"}, '
            f'Torch Compile: {"Enabled" if torch_compile_enabled else "Disabled"},', icon="ℹ️")
    
    return base

@st.cache_resource()
def load_image_to_image_model():
    # Load the base model for image-to-image
    base = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    base.to("cuda")
    st.toast('Base Model loading completed', icon="ℹ️")
    #pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    return base

@st.cache_resource()
def load_inpainting_model():
    # Load the base model for inpainting
    base = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    base.to("cuda")
    st.toast('Base Model loading completed', icon="ℹ️")
    #pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    return base

@st.cache_resource()
def load_refiner_model(_base, cpu_offload):
    # Load the refiner model
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=_base.text_encoder_2,
        vae=_base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    cpu_offload_enabled = False
    if cpu_offload:
        refiner.enable_model_cpu_offload()
        cpu_offload_enabled = True
    else:
        refiner.to("cuda")

    torch_compile_enabled = False
    if os.name == 'posix':
        refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
        torch_compile_enabled = True

    st.toast(f'Refiner Model loading completed. CPU Offload: {"Enabled" if cpu_offload_enabled else "Disabled"}, '
            f'Torch Compile: {"Enabled" if torch_compile_enabled else "Disabled"},', icon="ℹ️")

    return refiner


def text_to_image(prompt, negative_prompt, height, width, num_steps, high_noise_frac, guidance_scale, use_refiner):
    if use_refiner:
        image = st.session_state.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height = height,
            width = width,
            num_inference_steps=num_steps,
            denoising_end=high_noise_frac,
            guidance_scale=guidance_scale,
            output_type="latent",
        ).images
        
        image = st.session_state.refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height = height,
            width = width,
            num_inference_steps=num_steps,
            denoising_start=high_noise_frac,  # Changed from denoising_end
            guidance_scale=guidance_scale,
            image=image,  # Added this line
        ).images[0]
        st.write(f"Result: {image}")
        return image
    
    elif not use_refiner:
        image = st.session_state.base(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height = height,
        width = width,
        num_inference_steps=num_steps,
        denoising_end=high_noise_frac,
        guidance_scale=guidance_scale,
        output_type="latent",
        ).images[0]
        st.write(f"Result: {image}")
        return image

    elif image is None:
        st.error("Failed to refine image. Please contact the developer.")
        return None



def image_to_image(prompt, negative_prompt,num_steps,high_noise_frac,guidance_scale,init_image,use_refiner):
    # If init_image is None, display an error
    if init_image is None:
        st.warning("Please upload an image for image-to-image mode.")
        return None
    image = st.session_state.base(
        prompt, 
        negative_prompt=negative_prompt,
        image=init_image).images[0]
    
    if use_refiner:
        image = st.session_state.refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            denoising_start=high_noise_frac,  # Changed from denoising_end
            guidance_scale=guidance_scale,
            image=image,  # Added this line
        ).images[0]
        st.write(f"Result: {image}")
        
        if image is None:
            st.error("Failed to refine image.")
            return None
 
        return image
    
    if not use_refiner:
        st.write(f"Result: {image}")
        return image[0]
    return image

def inpainting(prompt, init_image, mask_image,num_inference_steps,high_noise_frac,use_refiner):
    if init_image is None or mask_image is None:
        st.warning("Please upload both an image and a mask for inpainting mode.")
        return None
    image = st.session_state.base(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=num_inference_steps,
        denoising_start=high_noise_frac,
        output_type="latent",
    ).images

    if use_refiner:
        image = st.session_state.refiner(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
        ).images[0]
        return image
    return image[0]


def handle_text2img(prompt, negative_prompt, height, width, num_steps, high_noise_frac, guidance_scale):
    use_refiner = st.sidebar.checkbox('Use refiner', value=True)
    if st.sidebar.button("Generate"):
        image = text_to_image(prompt, negative_prompt, height, width,num_steps, high_noise_frac, guidance_scale, use_refiner)
        st.image(image)



def handle_img2img(prompt, negative_prompt, num_steps, high_noise_frac, guidance_scale):
    init_image_file = st.sidebar.file_uploader("Upload an initial image", type=["jpg", "png"])
    if init_image_file is None:
        st.warning("Please upload an image for image-to-image mode.")
        return

    if st.sidebar.button("Generate"):
        init_image = Image.open(io.BytesIO(init_image_file.read()))
        image = image_to_image(prompt, negative_prompt, num_steps, high_noise_frac,guidance_scale,init_image)
        display_image(image)


def handle_inpainting(prompt, negative_prompt, num_steps, denoising_end, guidance_scale):
    init_image_file = st.sidebar.file_uploader("Upload an initial image", type=["jpg", "png"])
    mask_image_file = st.sidebar.file_uploader("Upload a mask image", type=["jpg", "png"])
    if init_image_file is None or mask_image_file is None:
        st.warning("Please upload both an image and a mask for inpainting mode.")
        return

    if st.sidebar.button("Generate"):
        init_image = Image.open(io.BytesIO(init_image_file.read()))
        mask_image = Image.open(io.BytesIO(mask_image_file.read()))
        image = inpainting(prompt, negative_prompt, init_image, mask_image, num_steps, denoising_end, guidance_scale)
        display_image(image)



def save_image(image, save_folder=None, image_name=None):
    """
    Save the given image to the specified folder.
    If the folder doesn't exist, it creates one.
    The image is saved with a timestamp-based name if no name is provided.
    
    :param image: Image to save (PIL Image object)
    :param save_folder: Folder to save the image. Defaults to "saved_images".
    :param image_name: Name of the saved image. Defaults to a timestamp.
    """
    if save_folder is None:
        save_folder = "saved_images"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # If no image_name is provided, use a timestamp
    if image_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"image_{timestamp}.png"

    image_path = os.path.join(save_folder, image_name)
    image.save(image_path, "PNG")
    print(f"Image saved at: {image_path}")



def display_image(image):
    if image is not None:
        st.image(image)
