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

POSITIVE_PROMPT_OPTIONS =  [
    "beautiful", "stunning", "breathtaking", "ethereal", "radiant", "captivating", "mesmerizing", "alluring", 
    "elegant", "serene", "luminous", "vibrant", "intricate", "masterful", "impressive", "extraordinary", 
    "remarkable", "exceptional", "flawless", "exquisite"
]

selected_negative_prompts = ["NSFW","low-res","bad anatomy","text","error","cropped","low quality","blurry","watermark","bad proportions"]

selected_positive_prompts = [
    "Breathtaking",
    "Captivating",
    "Elegant",
    "Mesmerizing",
    "Serene",
    "Vibrant",
    "Masterful",
    "Extraordinary",
    "Exquisite",
    "Awe-inspiring",
    "8K",
    "Masterpiece",
]

def sidebar():
    pass
def text_to_image():
    # Sidebar setup
    st.sidebar.title("Parameters")
    global selected_negative_prompts
    # Rest of the prompt and parameter setup
    default_prompt = "realism of a built, angular, Nervous Traditionalist woman, Stately, :o, with long black hair wearing cardigan vest , digital art, highly detailed, fine detail, intricate, outrun, vaporware"
    prompt1 = st.sidebar.text_area(label="Enter a prompt", value=default_prompt, height=200)
    prompt2_1 = st.sidebar.text_area(label="Write Auxiliary prompt here...", placeholder=" (Optional)", height=100)
    prompt2_2 = st.sidebar.multiselect("Or select Auxiliary Positive Prompts", selected_positive_prompts, default=selected_positive_prompts)
    prompt2 = prompt2_1 + ", ".join(prompt2_2)  # Convert the list of selected options into a string
    height = st.sidebar.slider("Height", 512, 1024, 1024, step=8)
    width = st.sidebar.slider("Width", 512, 1024, 1024, step=8)
    num_steps = st.sidebar.slider("Number of steps", 10, 100, 40)
    denoising_end = st.sidebar.slider("Denoising end fraction", 0.0, 1.0, 0.8)
    guidance_scale = st.sidebar.slider("Guidance scale", 1.0, 10.0, 7.5)
    selected_negative_prompts = st.sidebar.multiselect("Negative Prompts", NEGATIVE_PROMPT_OPTIONS, default=selected_negative_prompts)
    negative_prompt = ", ".join(selected_negative_prompts)  # Convert the list of selected options into a string
    analytics = {
            "prompt1": prompt1,
            "prompt2": prompt2,
            "negative_prompt": negative_prompt,
            "h": height,
            "w": width,
            "num_steps": num_steps,
            "high_noise_frac": denoising_end,
            "guidance_scale": guidance_scale
        }
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
        st.toast(response)

        if response.status_code == 200:
            st.write("Image generated successfully.")
            image_base64 = response.json().get("image_base64")
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
            st.image(image, caption="Generated Image", use_column_width=True)
            st.write(analytics)
        else:
            st.warning("Failed to generate image.")
            st.write(response.text)


def image_to_image():
    pass

def image_to_text():
    pass

def ControlNet():
    pass

def YOLOv8_Anime():
    pass

def My_Account():
    pass

def About():
    st.write("This is a web app for the SDXL 1.0 model. It is a work in progress.")

def main():

    tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs(["Text to Image", "Image to Image", "Image to Text","ControlNet","YOLOv8 Anime","My Account","About"])
    # Page info
    st.set_page_config(
        page_title="SDXL 1.0 Web App",
        page_icon=":robot:",
        layout='wide'
    )

    with tab1:
        text_to_image()
    
    with tab2:
        image_to_image()

if __name__ == "__main__":
    main()