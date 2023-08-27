from fastapi import FastAPI
from pydantic import BaseModel
import base64
from io import BytesIO
from cuda import cudart
import tensorrt as trt
from utilities import TRT_LOGGER
from txt2img_xl_pipeline import Txt2ImgXLPipeline
from img2img_xl_pipeline import Img2ImgXLPipeline
from PIL import Image

class ImageRequest(BaseModel):
    prompt: str
    h: int
    w: int
    num_steps: int

app = FastAPI()

# Arguments for initializing the pipeline
args = {
    'scheduler': 'DPM',
    'denoising_steps': 30,
    'output_dir': './output',
    'version': 'xl-1.0',
    'hf_token': None,
    'verbose': False,
    'nvtx_profile': False,
    'max_batch_size': 16,
    'use_cuda_graph': True,
    'framework_model_dir': './model_dir',
    'onnx_opset': 12,
    'height': 1024,
    'width': 1024,
    'force_onnx_export': False,
    'force_onnx_optimize': False,
    'force_engine_build': False,
    'build_static_batch': True,
    'build_dynamic_shape': False,
    'build_enable_refit': False,
    'build_preview_features': False,
    'build_all_tactics': False,
    'timing_cache': None,
    'onnx_refit_dir': None,
    'seed': 0,
    'num_warmup_runs': 1,
    'onnx_base_dir': '/mnt/h/SD_TRT/stable-diffusion-xl-1.0-tensorrt/sdxl-1.0-base',
    'onnx_refiner_dir': '/mnt/h/SD_TRT/stable-diffusion-xl-1.0-tensorrt/sdxl-1.0-refiner',
}

# Initialize TensorRT SDXL Pipeline at startup
def init_sdxl_pipeline(pipeline_class, refiner, onnx_dir, engine_dir, args):
    demo = pipeline_class(
        scheduler=args['scheduler'],
        denoising_steps=args['denoising_steps'],
        output_dir=args['output_dir'],
        version=args['version'],
        hf_token=args['hf_token'],
        verbose=args['verbose'],
        nvtx_profile=args['nvtx_profile'],
        max_batch_size=args['max_batch_size'],
        use_cuda_graph=args['use_cuda_graph'],
        refiner=refiner,
        framework_model_dir=args['framework_model_dir']
    )
    demo.loadEngines(engine_dir, args['framework_model_dir'], onnx_dir, args['onnx_opset'],
                     opt_batch_size=1,  # This can be modified based on your use-case
                     opt_image_height=args['height'],
                     opt_image_width=args['width'],
                     force_export=args['force_onnx_export'],
                     force_optimize=args['force_onnx_optimize'],
                     force_build=args['force_engine_build'],
                     static_batch=args['build_static_batch'],
                     static_shape=not args['build_dynamic_shape'],
                     enable_refit=args['build_enable_refit'],
                     enable_preview=args['build_preview_features'],
                     enable_all_tactics=args['build_all_tactics'],
                     timing_cache=args['timing_cache'],
                     onnx_refit_dir=args['onnx_refit_dir']
                     )
    return demo

def run_sd_xl_inference(prompt, negative_prompt, image_height, image_width, warmup=False, verbose=False):
    images, time_base = demo_base.infer(prompt, negative_prompt, image_height, image_width, warmup=warmup, verbose=verbose, seed=args.seed, return_type="latents")
    images, time_refiner = demo_refiner.infer(prompt, negative_prompt, images, image_height, image_width, warmup=warmup, verbose=verbose, seed=args.seed)
    return images, time_base + time_refiner

demo_base = init_sdxl_pipeline(Txt2ImgXLPipeline, False, args['onnx_base_dir'], 'engine_base_dir', args)
demo_refiner = init_sdxl_pipeline(Img2ImgXLPipeline, True, args['onnx_refiner_dir'], 'engine_refiner_dir', args)
max_device_memory = max(demo_base.calculateMaxDeviceMemory(), demo_refiner.calculateMaxDeviceMemory())
_, shared_device_memory = cudart.cudaMalloc(max_device_memory)
demo_base.activateEngines(shared_device_memory)
demo_refiner.activateEngines(shared_device_memory)
demo_base.loadResources(args["image_height"], args["image_width"], args["batch_size"], args["seed"])
demo_refiner.loadResources(args["image_height"], args["image_width"], args["batch_size"], args["seed"])

prompt = "A photo of a cat"
negative_prompt = "A photo of a dog"

if args["use_cuda_graph"]:
    # inference once to get cuda graph
    images, _ = run_sd_xl_inference(prompt, negative_prompt, 1024,1024,warmup=True, verbose=False)

print("[I] Warming up ..")
for _ in range(args["num_warmup_runs"]):
    images, _ = run_sd_xl_inference(prompt, negative_prompt, 1024,1024,warmup=True, verbose=False)

print("[I] Running StableDiffusion pipeline")
# if args.nvtx_profile:
#     cudart.cudaProfilerStart()
# images, pipeline_time = run_sd_xl_inference(warmup=False, verbose=args.verbose)
# if args.nvtx_profile:
#     cudart.cudaProfilerStop()




@app.post("/generate-and-refine/")
async def generate_and_refine_image(request: ImageRequest):
    # Process prompt
    prompt = [request.prompt1+request.prompt2]
    negative_prompt = [request.negative_prompt]
    image_height = request.h
    image_width = request.w
    
    # Run the SDXL TensorRT inference
    image, _ = run_sd_xl_inference(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        image_height=image_height, 
        image_width=image_width, 
        warmup=False, 
        verbose=False
    )
    
    # Convert the generated image to JPEG and base64 encode it
    image = Image.fromarray(images[0].astype('uint8'))  # Assuming the image is in uint8 format
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {"image_base64": img_str}




