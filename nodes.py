import os
import torch
import sys
sys.path.append("../..")

from diffusers.pipelines import FluxPipeline
from src.condition import Condition
from PIL import Image
import safetensors.torch
from src.generate import generate, seed_everything
import comfy.sd
import folder_paths

# Memory optimization settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
#model, clip, vae = out[:3]

# Memory optimization settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#torch.backends.cudnn.benchmark = True

def setup_pipeline(ckpt_path):
    pipe = FluxPipeline.from_single_file(
        ckpt_path,
        torch_dtype=torch.float8_e4m3fn,
        low_cpu_mem_usage=True
    )
    
    # Enable sequential CPU offload properly
    #pipe.enable_sequential_cpu_offload(gpu_id=0)
    pipe.enable_model_cpu_offload()
    
    # Enable memory efficient attention
    pipe.enable_attention_slicing(slice_size="auto")
    pipe.enable_vae_slicing()
    
    #pipe.vae.to(dtype=torch.float16)
    pipe.text_encoder.to(dtype=torch.float16)
    pipe.text_encoder_2.to(dtype=torch.float16)
    
    # Load LoRA weights after CPU offloading is set up
    pipe.load_lora_weights(
        "Yuanshi/OminiControl",
        weight_name="omini/subject_512.safetensors",
        adapter_name="subject",
    )

    torch.cuda.empty_cache()
    print("Model loaded and optimized!")
    
    return pipe

def generate_image(pipe, image_path, prompt, seed=0):
    # Load and prepare input image
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    condition = Condition("subject", image)
    
    # Set seed for reproducibility
    seed_everything(seed)
    
    with torch.inference_mode():
        result_img = generate(
            pipe,
            prompt=prompt,
            conditions=[condition],
            num_inference_steps=8,
            height=512,
            width=512,
            guidance_scale=7.5,
        ).images[0]
        
    return image, result_img

def main():
    ckpt_path = "/home/rednax/SSD2TB/Github_repos/ComfyUI/models/checkpoints/flux1-schnell-fp8.safetensors"
    prompt = "On Christmas evening, on a crowded sidewalk, this item sits on the road, covered in snow and wearing a Christmas hat."
    conditioning_image = "assets/penguin.jpg"
    
    pipe = setup_pipeline(ckpt_path)
    
    # Generate image
    original_img, result_img = generate_image(pipe, conditioning_image, prompt)
    
    # Combine images
    concat_image = Image.new("RGB", (1024, 512))
    concat_image.paste(original_img, (0, 0))
    concat_image.paste(result_img, (512, 0))
    
    return concat_image

if __name__ == "__main__":
    result = main()
    result.save("output.png")