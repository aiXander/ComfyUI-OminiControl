import modal
import os
import json
import os
import pathlib
import requests
from tqdm import tqdm

from example_subject import test_omini_control

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

def download_files(downloads_json_path: str, base_path: str = "/root"):
    """Download all files specified in a downloads.json file."""
    downloads = json.load(open(downloads_json_path, 'r'))
    
    for path, url in downloads.items():
        dest_path = pathlib.Path(base_path) / path
        print(f"Downloading {url} to {dest_path}")
        download_file(url, dest_path)



app = modal.App("OminiControl")

# Create a volume to cache downloads
downloads_vol = modal.Volume.from_name(
    "downloads-cache",
    create_if_missing=True
)

@modal.build()
def download_files_to_image():
    # Create workspace directory if it doesn't exist
    os.makedirs("/workspace", exist_ok=True)
    
    # Download files before running the main task
    download_files("/workspace/downloads.json", base_path="/data")

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
        "tqdm"
    )
)

@app.function(
    gpu="T4",
    image=image,
    volumes={"/data": downloads_vol},
    mounts=[modal.Mount.from_local_file("downloads.json", "/workspace/downloads.json")]
)
def run_omini():
    import subprocess

    print("here's my gpu:")
    try:
        subprocess.run(["nvidia-smi", "--list-gpus"], check=True)
    except Exception:
        print("no gpu found :(")

    test_omini_control()

@app.local_entrypoint()
def main():
    run_omini.remote()