import modal
import os
import json
import pathlib
import requests
from tqdm import tqdm
from huggingface_hub import snapshot_download

from example_subject import test_omini_control

def download_model_from_hf(repo_id: str, dest_path: str) -> str:
    """Download a complete model from HuggingFace to a destination path."""
    dest_path = pathlib.Path(dest_path)
    if dest_path.exists():
        print(f"Model already exists at {dest_path}, skipping download")
        return str(dest_path)
    
    print(f"Downloading model from {repo_id} to {dest_path}")
    snapshot_download(
        repo_id,
        local_dir=dest_path
    )
    return str(dest_path)

def download_file(url: str, dest_path: str) -> str:
    """Download a file from a URL to a destination path with progress bar."""
    dest_path = pathlib.Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists():
        print(f"File already exists at {dest_path}, skipping download")
        return str(dest_path)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))
            
    return str(dest_path)

app = modal.App("OminiControl")

downloads_vol = modal.Volume.from_name(
    "downloads-cache",
    create_if_missing=True
)

# Define the image with all necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "git-lfs", "libgl1-mesa-glx", "libglib2.0-0", "libegl1", 
        "libmagic1", "ffmpeg", "libsm6", "libxext6"
    )
    .pip_install(
        "transformers",
        "diffusers",
        "peft",
        "opencv-python",
        "protobuf",
        "sentencepiece",
        "gradio",
        "jupyter",
        "torchao",
        "requests",
        "tqdm",
        "accelerate",
        "safetensors",
        "huggingface_hub" 
    )
    .run_commands("mkdir -p /root/workspace/assets")
    .copy_local_file("downloads.json", "/root/workspace/downloads.json")
    .copy_local_file("assets/penguin.jpg", "/root/workspace/assets/penguin.jpg")
)

def print_system_stats():
    import subprocess
    print("System stats:")
    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(["lscpu"], check=True)


@app.function(
    gpu="T4",
    cpu=8.0,
    image=image,
    volumes={"/data": downloads_vol},
    concurrency_limit=3,
    container_idle_timeout=60,
    timeout=3600
)
def run_omini():
    print_system_stats()

    model_path = "/data/flux-schnell-hf-model"
    # Download complete model if needed
    if not os.path.exists(model_path):
        try:
            download_model_from_hf("black-forest-labs/FLUX.1-schnell", model_path)
            downloads_vol.commit()
        except Exception as e:
            print(f"Error downloading model: {e}")
            return

    # Load and process downloads
    try:
        with open("/root/workspace/downloads.json", 'r') as f:
            downloads = json.load(f)
        print("Downloads configuration:", downloads)
    except Exception as e:
        print(f"Error loading downloads.json: {e}")
        return

    # Set up model path
    pretrained_flux_path = "/data/flux1-schnell-fp8-e4m3fn.safetensors"
    
    # Download model if needed
    if not os.path.exists(pretrained_flux_path):
        print(f"Model file not found at {pretrained_flux_path}")
        model_url = downloads.get('flux1-schnell-fp8-e4m3fn.safetensors')
        if model_url:
            try:
                download_file(model_url, pretrained_flux_path)
                downloads_vol.commit()
            except Exception as e:
                print(f"Error downloading model: {e}")
                return
        else:
            print("Model URL not found in downloads.json")
            return

    test_omini_control(model_path)


@app.local_entrypoint()
def main():
    run_omini.remote()