# streamlit_frontend.py

import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

# Common negative prompt options
NEGATIVE_PROMPT_OPTIONS = [
    "NSFW", "Multiple people", "low-res", "bad anatomy", "bad hands", "text", "error", "missing fingers",
    "extra digit", "fewer digits", "cropped", "worst quality", "low quality", "normal quality", "jpeg artifacts",
    "signature", "watermark", "username", "blurry", "bad feet", "mutation", "deformed", "extra fingers",
    "fewer digits", "extra arms", "extra legs", "malformed limbs", "fused fingers", "too many fingers", "long neck",
    "cross-eyed", "mutated hands", "polar low-res", "bad body", "bad proportions", "gross proportions"
]

def main():
    # Page info
    st.set_page_config(
        page_title="SDXL 1.0 Web App",
        page_icon=":robot:",
        layout='wide'
    )

    # Sidebar setup
    st.sidebar.title("Parameters")
    mode = st.sidebar.selectbox("Mode", ["text2img", "img2img", "inpainting"], index=0)
    
    # Check if the mode has changed and set the appropriate mode in session state
    if "mode" not in st.session_state or st.session_state.mode != mode:
        st.session_state.mode = mode

    # Rest of the prompt and parameter setup
    default_prompt = "Japanese girl in a red dress is walking in the forest, looking at the viewer"
    prompt1 = st.sidebar.text_area(label="Enter a prompt", value=default_prompt, height=200)
    prompt2 = st.sidebar.text_area(label="Auxiliary prompt", placeholder=" (Not necessary)", height=200)
    height = st.sidebar.slider("Height", 128, 1024, 512)
    width = st.sidebar.slider("Width", 128, 1024, 512)
    num_steps = st.sidebar.slider("Number of steps", 10, 100, 50)
    denoising_end = st.sidebar.slider("Denoising end fraction", 0.0, 1.0, 0.8)
    guidance_scale = st.sidebar.slider("Guidance scale", 1.0, 10.0, 7.5)
    selected_negative_prompts = st.sidebar.multiselect("Negative Prompts", NEGATIVE_PROMPT_OPTIONS, default=[
        "NSFW", "Multiple people", "low-res", "bad anatomy", "bad hands", "text", "error", "missing fingers"])
    negative_prompt = ", ".join(selected_negative_prompts)  # Convert the list of selected options into a string

    # Button to trigger image generation
    if st.sidebar.button("Generate Image"):
        response = requests.post("http://localhost:8000/generate-and-refine/", json={
            "prompt1": prompt1,
            "prompt2": prompt2,
            "negative_prompt": negative_prompt,
            "h": height,
            "w": width,
            "num_steps": num_steps,
            "high_noise_frac": denoising_end,
            "guidance_scale": guidance_scale
        })

        if response.status_code == 200:
            image_base64 = response.json().get("image_base64")
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
            st.image(image, caption="Generated Image", use_column_width=True)
        else:
            st.warning("Failed to generate image.")
            st.write(response.text)


if __name__ == "__main__":
    main()
