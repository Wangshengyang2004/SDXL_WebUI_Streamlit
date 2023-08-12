import streamlit as st
from diffusers import DiffusionPipeline
from PIL import Image

# Initialization
@st.cache_resource
def load_model():
    return DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

model = load_model()

# Web UI
st.title("Text-to-Image Generator")

# Collecting inputs from the user
prompt = st.text_area("Enter your prompt", "A majestic lion jumping from a big stone at night")
prompt_2 = st.text_area("Enter your secondary prompt (optional)", "")
height = st.slider("Height", 128, 1024, 512)
width = st.slider("Width", 128, 1024, 512)
num_steps = st.slider("Number of steps", 10, 100, 50)
denoising_end = st.slider("Denoising end fraction", 0.0, 1.0, 0.8)
guidance_scale = st.slider("Guidance scale", 1.0, 10.0, 7.5)
negative_prompt = st.text_area("Negative prompt (optional)", "")
negative_prompt_2 = st.text_area("Negative secondary prompt (optional)", "")
num_images_per_prompt = st.slider("Number of images per prompt", 1, 10, 1)
eta = st.slider("Eta", 0.0, 1.0, 0.0)
guidance_rescale = st.slider("Guidance rescale", 0.1, 1.0, 0.7)
original_size = tuple(st.slider("Original size (height, width)", 128, 1024, (512, 512), 2))
crops_coords_top_left = tuple(st.slider("Crops coordinates top left (x, y)", 0, 1024, (0, 0), 2))
target_size = tuple(st.slider("Target size (height, width)", 128, 1024, (512, 512), 2))

if st.button("Generate Image"):
    image = model(
        prompt=prompt,
        prompt_2=prompt_2 if prompt_2 else None,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        denoising_end=denoising_end,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt if negative_prompt else None,
        negative_prompt_2=negative_prompt_2 if negative_prompt_2 else None,
        num_images_per_prompt=num_images_per_prompt,
        eta=eta,
        guidance_rescale=guidance_rescale,
        original_size=original_size,
        crops_coords_top_left=crops_coords_top_left,
        target_size=target_size,
        output_type="pil"
    )
    st.image(image.images[0], caption="Generated Image", use_column_width=True)

