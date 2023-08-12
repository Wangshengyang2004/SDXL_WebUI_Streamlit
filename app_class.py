import streamlit as st
from diffusers.utils import load_image
from utilsclass import SDXL_Pipeline  # Ensure SDXL_Pipeline is imported

# Common negative prompt options
NEGATIVE_PROMPT_OPTIONS = ["NSFW", "Multiple people", "low-res", "bad anatomy", "bad hands", "text", "error", "missing fingers",
                           "extra digit", "fewer digits", "cropped", "worst quality", "low quality", "normal quality", "jpeg artifacts",
                           "signature", "watermark", "username", "blurry", "bad feet", "mutation", "deformed", "extra fingers",
                           "fewer digits", "extra arms", "extra legs", "malformed limbs", "fused fingers", "too many fingers", "long neck",
                           "cross-eyed", "mutated hands", "polar low-res", "bad body", "bad proportions", "gross proportions"]

def main():
    # Page info
    st.set_page_config(
        page_title="SDXL 1.0 Web App",
        page_icon=":robot:",
        layout='wide'
    )

    # Instantiate the SDXL_Pipeline class
    pipeline = SDXL_Pipeline()

    # Sidebar setup
    st.sidebar.title("Parameters")
    mode = st.sidebar.selectbox("Mode", ["text2img", "img2img", "inpainting"], index=0)
    cpu_offload = st.sidebar.checkbox('Enable CPU Offload', value=False)

    # If the CPU offload checkbox states have changed, clear the pipeline and reload the models
    if ("cpu_offload" not in st.session_state or st.session_state.cpu_offload != cpu_offload):
        st.session_state.cpu_offload = cpu_offload
        st.session_state.base = None
        st.session_state.refiner = None

    # Check if the mode has changed. If it has, clear the pipeline and reload the model.
    if "mode" not in st.session_state or st.session_state.mode != mode:
        st.session_state.mode = mode
        st.session_state.base = None
        st.session_state.refiner = None

    # Load models using the pipeline methods
    if st.session_state.base is None:
        if mode == "text2img":
            pipeline.load_base_model(cpu_offload)
        elif mode == "img2img":
            pipeline.load_image_to_image_model()
        elif mode == "inpainting":
            pipeline.load_inpainting_model()

    if st.session_state.refiner is None:
        pipeline.load_refiner_model(cpu_offload)

    # Rest of the prompt and parameter setup
    default_prompt = "Japanese girl in a red dress is walking in the forest, looking at the viewer"
    prompt1 = st.sidebar.text_area(label="Enter a prompt", value=default_prompt, height=200)
    prompt2 = st.sidebar.text_area(label="Auxilary prompt", placeholder=" (Not necessay)",height=200)
    height = st.sidebar.slider("Height", 128, 1024, 512)
    width = st.sidebar.slider("Width", 128, 1024, 512)
    num_steps = st.sidebar.slider("Number of steps", 10, 100, 50)
    denoising_end = st.sidebar.slider("Denoising end fraction", 0.0, 1.0, 0.8)
    guidance_scale = st.sidebar.slider("Guidance scale", 1.0, 10.0, 7.5)
    selected_negative_prompts = st.sidebar.multiselect("Negative Prompts", NEGATIVE_PROMPT_OPTIONS, default=["NSFW", "Multiple people", "low-res", "bad anatomy", "bad hands", "text", "error", "missing fingers"])
    negative_prompt = ", ".join(selected_negative_prompts)  # Convert the list of selected options into a string

    base_image = pipeline.base_generate(prompt1=prompt1,
                           prompt2=prompt2,
                           negative_prompt=negative_prompt,
                           h=height,
                           w=width,
                           num_steps=num_steps,
                           high_noise_frac=denoising_end,
                           guidance_scale=guidance_scale
                           )

if __name__ == "__main__":
    main()
