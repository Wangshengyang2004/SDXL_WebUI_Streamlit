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
import torch

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str
    h: int
    w: int
    num_steps: int

app = FastAPI()

# Arguments for initializing the pipeline
args = {
    'scheduler': 'DDIM',
    'denoising_steps': 30,
    'output_dir': './output',
    'version': 'xl-1.0',
    'hf_token': None,
    'verbose': False,
    'nvtx_profile': True,
    'use_cuda_graph': True,
    'framework_model_dir': './model_dir',
    'onnx_opset': 12,
    'image_height': 1024,
    'image_width': 1024,
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
        max_batch_size=4,
        use_cuda_graph=args['use_cuda_graph'],
        refiner=refiner,
        framework_model_dir=args['framework_model_dir']
    )
    demo.loadEngines(engine_dir, args['framework_model_dir'], onnx_dir, args['onnx_opset'],
                     opt_batch_size=1,  # This can be modified based on your use-case
                     opt_image_height=args['image_height'],
                     opt_image_width=args['image_width'],
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
    images, time_base = demo_base.infer(prompt, negative_prompt, image_height, image_width, warmup=warmup, verbose=verbose, return_type="latents")
    images, time_refiner = demo_refiner.infer(prompt, negative_prompt, images, image_height, image_width, warmup=warmup, verbose=verbose)
    return images, time_base + time_refiner

def tensor_to_base64(image_tensor):
    # Squeeze out the batch dimension and convert tensor to uint8 numpy array
    image_tensor = image_tensor.squeeze(0)  # Remove batch dimension, now it becomes [3, 1024, 1024]
    image_np = ((image_tensor + 1) * 255 / 2).clamp(0, 255).detach().permute(1, 2, 0).round().type(torch.uint8).cpu().numpy()
    
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_np)
    
    # Save PIL Image to BytesIO object
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    
    # Base64 encode and decode to string
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str



demo_base = init_sdxl_pipeline(Txt2ImgXLPipeline, False, args['onnx_base_dir'], 'engine_xl_base',args)
demo_refiner = init_sdxl_pipeline(Img2ImgXLPipeline, True, args['onnx_refiner_dir'], 'engine_xl_refiner',args)
max_device_memory = max(demo_base.calculateMaxDeviceMemory(), demo_refiner.calculateMaxDeviceMemory())
_, shared_device_memory = cudart.cudaMalloc(max_device_memory)
demo_base.activateEngines(shared_device_memory)
demo_refiner.activateEngines(shared_device_memory)
image_height = 1024
image_width = 1024
batch_size = 1
demo_base.loadResources(image_height, image_width, batch_size,seed=0)
demo_refiner.loadResources(image_height, image_width, batch_size,seed=0)
test_prompt = ["A photo of a cat"]
test_negative_prompt = ["A photo of a dog"]
# FIXME VAE build fails due to element limit. Limitting batch size is WAR
if args["build_dynamic_shape"] or args["image_height"] > 512 or args["image_width"] > 512:
    max_batch_size = 4
if batch_size > max_batch_size:
    raise ValueError(f"Batch size {len(test_prompt)} is larger than allowed {max_batch_size}. If dynamic shape is used, then maximum batch size is 4")

if args["use_cuda_graph"] and (not args["build_static_batch"] or args["build_dynamic_shape"]):
    raise ValueError(f"Using CUDA graph requires static dimensions. Enable `--build-static-batch` and do not specify `--build-dynamic-shape`")

if args["use_cuda_graph"]:
# inference once to get cuda graph
    print("Use cuda graph")
    images, _ = run_sd_xl_inference(test_prompt, test_negative_prompt, 1024,1024,warmup=True, verbose=False)

print("[I] Warming up ..")
for _ in range(args["num_warmup_runs"]):
    images, _ = run_sd_xl_inference(test_prompt, test_negative_prompt, 1024,1024,warmup=True, verbose=False)
    type(images)
    print(images.shape)

@app.post("/generate-and-refine/")
async def generate_and_refine_image(request: ImageRequest):
    # Process prompt
    prompt = [request.prompt]
    negative_prompt = [request.negative_prompt]
    image_height = request.h
    image_width = request.w
    batch_size = 1

    print("[I] Running StableDiffusion pipeline")

    # Run the SDXL TensorRT inference
    if args["nvtx_profile"]:
        cudart.cudaProfilerStart()

    image, pipeline_time = run_sd_xl_inference(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        image_height=image_height, 
        image_width=image_width, 
        warmup=False, 
        verbose=False
    )
    if args["nvtx_profile"]:
        cudart.cudaProfilerStop()

    print('|------------|--------------|')
    print('| {:^10} | {:>9.2f} ms |'.format('e2e', pipeline_time))
    print('|------------|--------------|')

    # Convert the generated image to JPEG and base64 encode it
    img_str = tensor_to_base64(image)
    
    return {"image_base64": img_str}




