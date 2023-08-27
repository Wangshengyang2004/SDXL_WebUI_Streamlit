import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

st.title("Stable Diffusion XL")

st.write("""
### Generate and Refine Images
""")

# Create form
with st.form("generate_and_refine_form"):
    prompt1 = st.text_input("Enter a prompt:", "A photo of a cat")
    prompt2 = st.text_input("Enter a prompt:", "8K")
    h = st.number_input("Image Height:", 1024)
    w = st.number_input("Image Width:", 1024)
    

    submitted = st.form_submit_button("Generate and Refine")

# Call FastAPI endpoint
if submitted:
    # Define the API endpoint
    url = "http://127.0.0.1:8000/generate-and-refine/"

    # Create data payload
    data = {
        "prompt1": prompt1,
        "prompt2": prompt2,
        "h": h,
        "w": w,
    }

    # Make the API request
    response = requests.post(url, json=data)

    # Decode the base64 image
    if response.status_code == 200:
        result = response.json()
        image_base64 = result.get("image_base64")
        img = Image.open(BytesIO(base64.b64decode(image_base64)))

        # Display the image
        st.image(img, caption="Generated and Refined Image", use_column_width=True)
    else:
        st.write(f"Failed to generate image: {response.content}")
