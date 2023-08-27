import os
from cuda import cudart
import tensorrt as trt
import torch
from utilities import TRT_LOGGER
from txt2img_xl_pipeline import Txt2ImgXLPipeline
from img2img_xl_pipeline import Img2ImgXLPipeline

if __name__ == "__main__":
    print("[I] Initializing TensorRT accelerated StableDiffusionXL txt2img pipeline")
    
    # Set default values for arguments
    version = "xl-1.0"
    height = 1024
    width = 1024
    scheduler = "DDIM"
    onnx_base_dir = "/mnt/h/SD_TRT/stable-diffusion-xl-1.0-tensorrt/sdxl-1.0-base"
    print(os.listdir(onnx_base_dir))
    onnx_refiner_dir = "/mnt/h/SD_TRT/stable-diffusion-xl-1.0-tensorrt/sdxl-1.0-refiner"
    engine_base_dir = 'engine_xl_base'
    engine_refiner_dir = 'engine_xl_refiner'
    
    # Additional defaults (replace these with the actual defaults if known)
    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"  # Replace with your actual prompt
    negative_prompt = []  # Replace if necessary
    denoising_steps = 30  # Replace with default value
    output_dir = "./output"  # Replace with default value
    hf_token = None  # Replace with default value
    verbose = False  # Replace with default value
    nvtx_profile = False  # Replace with default value
    build_dynamic_shape = False  # Replace with default value
    use_cuda_graph = True  # Replace with default value
    force_onnx_export = False  # Replace with default value
    force_onnx_optimize = False  # Replace with default value
    force_engine_build = False  # Replace with default value
    build_static_batch = True  # Replace with default value
    build_enable_refit = False  # Replace with default value
    build_preview_features = False  # Replace with default value
    build_all_tactics = False  # Replace with default value
    seed = 114514  # Replace with default value
    num_warmup_runs = 1  # Replace with default value

    # Validate image dimensions
    image_height = height
    image_width = width
    if image_height % 8 != 0 or image_width % 8 != 0:
        raise ValueError(f"Image height and width have to be divisible by 8 but specified as: {image_height} and {image_width}.")

    # Register TensorRT plugins
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    max_batch_size = 16
    if build_dynamic_shape or image_height > 512 or image_width > 512:
        max_batch_size = 4

    batch_size = 1  # Since we're generating a single image based on the prompt
    if batch_size > max_batch_size:
        raise ValueError(f"Batch size {batch_size} is larger than allowed {max_batch_size}. If dynamic shape is used, then maximum batch size is 4")

    if use_cuda_graph and (not build_static_batch or build_dynamic_shape):
        raise ValueError(f"Using CUDA graph requires static dimensions. Enable `build_static_batch` and do not specify `build_dynamic_shape`")

        # Initialization and other code
    
    def init_pipeline(pipeline_class, refiner, onnx_dir, engine_dir):
        # Initialize demo
        demo = pipeline_class(
            scheduler=scheduler,
            denoising_steps=denoising_steps,
            output_dir=output_dir,
            version=version,
            hf_token=hf_token,
            verbose=verbose,
            nvtx_profile=nvtx_profile,
            max_batch_size=max_batch_size,
            use_cuda_graph=use_cuda_graph,
            refiner=refiner,
           )

        # Load TensorRT engines and PyTorch modules
        demo.loadEngines(engine_dir,  onnx_dir, 
            opt_batch_size=batch_size, opt_image_height=image_height, opt_image_width=image_width, 
            force_export=force_onnx_export, force_optimize=force_onnx_optimize,
            force_build=force_engine_build,
            static_batch=build_static_batch, static_shape=not build_dynamic_shape, 
            enable_refit=build_enable_refit, enable_preview=build_preview_features, 
            enable_all_tactics=build_all_tactics,
            )
        return demo

    demo_base = init_pipeline(Txt2ImgXLPipeline, False, onnx_base_dir, engine_base_dir)
    demo_refiner = init_pipeline(Img2ImgXLPipeline, True, onnx_refiner_dir, engine_refiner_dir)
    max_device_memory = max(demo_base.calculateMaxDeviceMemory(), demo_refiner.calculateMaxDeviceMemory())
    _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
    demo_base.activateEngines(shared_device_memory)
    demo_refiner.activateEngines(shared_device_memory)

    demo_base.loadResources(image_height, image_width, batch_size, seed)
    demo_refiner.loadResources(image_height, image_width, batch_size, seed)

    # Define the inference function (assumes same arguments as original code)
    def run_sd_xl_inference(warmup=False, verbose=False):
        images, time_base = demo_base.infer(prompt, negative_prompt, image_height, image_width, warmup=warmup, verbose=verbose, seed=seed, return_type="latents")
        images, time_refiner = demo_refiner.infer(prompt, negative_prompt, images, image_height, image_width, warmup=warmup, verbose=verbose, seed=seed)
        return images, time_base + time_refiner

    if use_cuda_graph:
        # Inference once to get CUDA graph
        images, _ = run_sd_xl_inference(warmup=True, verbose=False)

    print("[I] Warming up ..")
    for _ in range(num_warmup_runs):
        images, _ = run_sd_xl_inference(warmup=True, verbose=False)

    for _ in range(20):  # Generate the image 20 times
        print("[I] Running StableDiffusion pipeline")
        
        if nvtx_profile:
            cudart.cudaProfilerStart()
        
        images, pipeline_time = run_sd_xl_inference(warmup=False, verbose=verbose)  # Assuming run_sd_xl_inference is defined similarly
        
        if nvtx_profile:
            cudart.cudaProfilerStop()

        print('|------------|--------------|')
        print('| {:^10} | {:>9.2f} ms |'.format('e2e', pipeline_time))
        print('|------------|--------------|')
    
    demo_base.teardown()
    demo_refiner.teardown()
