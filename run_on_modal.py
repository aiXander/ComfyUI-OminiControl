import modal
import os
from example_subject import test
from download_helper import download_files

app = modal.App("OminiControl")

# Create a volume to cache downloads
downloads_vol = modal.Volume.from_name(
    "downloads-cache",
    create_if_missing=True
)

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
    
    # Create workspace directory if it doesn't exist
    os.makedirs("/workspace", exist_ok=True)
    
    # Download files before running the main task
    download_files("/workspace/downloads.json", base_path="/data")
    
    print("here's my gpu:")
    try:
        subprocess.run(["nvidia-smi", "--list-gpus"], check=True)
    except Exception:
        print("no gpu found :(")

    test_omini_control()

@app.local_entrypoint()
def main():
    run_omini.remote()