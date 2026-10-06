import os
import io
import random
import zipfile
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from cvat_sdk import make_client
from src.seeai.config.settings import CVAT_HOST_IP, CVAT_HOST_PORT, CVAT_USER, CVAT_PASSWORD

# Construct CVAT Host URL
CVAT_HOST = f"{CVAT_HOST_IP}:{CVAT_HOST_PORT}"

def get_format_pretrained_models(format_name=None):
    """
    Scans values in models/pre_trained/detection or segmentation for .pt files
    """
    # Default to detection
    sub_dir = "detection"
    
    if format_name:
        if "Segmentation" in format_name:
            sub_dir = "segmentation"
        elif "Classification" in format_name:
            sub_dir = "classification"
    
    # Models dir relative to AI_PROJECT root
    # parents[0] = all_utils, parents[1] = src/seeai, parents[2] = src, parents[3] = AI_PROJECT
    models_dir = Path(__file__).resolve().parents[3] / "data/models/pre_trained" / sub_dir
    if not models_dir.exists():
        return []
    
    return [f.name for f in models_dir.glob("*.pt")]

def get_model_plots(model_path):
    """
    Returns list of plot images from 1.Train/{project}/{model}/...
    Specific files: BoxF1_curve, BoxP_curve, BoxPR_curve, BoxR_curve,
    confusion_matrix_normalized, confusion_matrix, labels, results.
    """
    if not model_path: return []
    
    # model_path is like: .../1.Train/Project_126/yolo11n-cls_49/weights/best.pt
    # Plots are in: .../1.Train/Project_126/yolo11n-cls_49/
    # So go up 2 levels: best.pt -> weights -> yolo11n-cls_49
    model_dir = Path(model_path).parent.parent
    if not model_dir.exists(): return []
    
    targets = [
        "BoxF1_curve.png", "BoxP_curve.png", "BoxPR_curve.png", "BoxR_curve.png",
        "confusion_matrix_normalized.png", "confusion_matrix.png",
        "labels.jpg", "labels.png", "results.png", "results.jpg"
    ]
    
    plots = []
    for t in targets:
        p = model_dir / t
        if p.exists():
            plots.append((p.stem, str(p.resolve())))
            
    # Return list of (path, label) tuples for Gallery
    return [(path, label) for label, path in plots]

def get_random_sample_images(project_id, datasets_dir, count=4):
    """
    Extracts up to 'count' random images from the CVAT project zip.
    Downloads the zip if it doesn't exist.
    """
    if not project_id:
        return None
    
    # 1. Check/Download Zip
    raw_zip_path = Path(datasets_dir) / f"cvat_project_{project_id}.zip"
    
    if not raw_zip_path.exists():
        # Attempt download
        try:
            url = CVAT_HOST
            username = CVAT_USER
            password = CVAT_PASSWORD
            
            # Sanitize host
            host = url.split('/projects')[0].split('/tasks')[0].split('jobs')[0].rstrip('/')
            
            # Add protocol if missing
            if not host.startswith(('http://', 'https://')):
                 host = 'http://' + host

            print(f"Downloading Project {project_id} for preview...")
            with make_client(host, credentials=(username, password)) as client:
                project = client.projects.retrieve(int(project_id))
                raw_zip_path.parent.mkdir(parents=True, exist_ok=True)
                project.export_dataset(
                    format_name="Ultralytics YOLO Detection 1.0",
                    filename=str(raw_zip_path),
                    include_images=True
                )
        except Exception as e:
            print(f"Error downloading for preview: {e}")
            return None

    # 2. Extract Random Image
    try:
        with zipfile.ZipFile(raw_zip_path, 'r') as zip_ref:
            # Filter for images
            file_list = zip_ref.namelist()
            image_files = [f for f in file_list if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and 'images' in f]
            
            if not image_files:
                return []
                
            # Select up to 'count' distinct images
            count = min(count, len(image_files))
            random_files = random.sample(image_files, count)
            
            images = []
            for fname in random_files:
                with zip_ref.open(fname) as file:
                    img_data = file.read()
                    images.append(Image.open(io.BytesIO(img_data)))
            
            return images

    except Exception as e:
        print(f"Error extracting preview images: {e}")
        return []
