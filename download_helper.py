import json
import os
import pathlib
import requests
from tqdm import tqdm

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