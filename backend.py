from fastapi import FastAPI
from pydantic import BaseModel
import base64
from io import BytesIO
from backend_utils import SDXL_Pipeline

class ImageRequest(BaseModel):
    prompt1: str
    prompt2: str
    negative_prompt: str
    h: int
    w: int
    num_steps: int
    high_noise_frac: float
    guidance_scale: float

app = FastAPI()

# Initialize the pipeline and load the models at startup
pipeline = SDXL_Pipeline()
pipeline.load_base_model()
pipeline.load_refiner_model()

@app.post("/generate-and-refine/")
async def generate_and_refine_image(request: ImageRequest):
    refined_image = pipeline.generate_and_refine(
        prompt1=request.prompt1,
        prompt2=request.prompt2,
        negative_prompt=request.negative_prompt,
        h=request.h,
        w=request.w,
        num_steps=request.num_steps,
        high_noise_frac=request.high_noise_frac,
        guidance_scale=request.guidance_scale
    )
    
    buffered = BytesIO()
    refined_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {"image_base64": img_str}