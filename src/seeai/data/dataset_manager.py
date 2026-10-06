import os
import shutil
import time
import random
import yaml
from pathlib import Path
from .file_utils import extract_and_flatten_zip

def remove_dataset_files(dataset_path):
    """
    Removes a dataset directory and its associated zip file from the filesystem.
    Returns: (success, message)
    """
    if not dataset_path:
        return False, "No dataset path provided."

    dataset_path_obj = Path(dataset_path)
    
    # Remove directory
    if dataset_path_obj.exists() and dataset_path_obj.is_dir():
        try:
            shutil.rmtree(dataset_path_obj)
            dir_msg = f"Successfully deleted directory: {dataset_path}"
        except OSError as e:
            return False, f"Error deleting directory {dataset_path}: {e}"
    else:
        dir_msg = f"Directory {dataset_path} not found."

    # Remove zip
    zip_path = dataset_path_obj.with_suffix('.zip')
    if zip_path.exists():
        try:
            zip_path.unlink()
            zip_msg = f"Deleted zip: {zip_path}"
        except OSError as e:
            zip_msg = f"Could not delete zip {zip_path}: {e}"
    else:
        zip_msg = "No zip file found."
        
    return True, f"{dir_msg}. {zip_msg}"

def _sanitize_stats_for_json(data):
    """Recursively ensure all dictionary keys are strings for JSON compatibility."""
    if isinstance(data, dict):
        return {str(k): _sanitize_stats_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize_stats_for_json(i) for i in data]
    else:
        return data

def inspect_dataset_zip(zip_path, datasets_dir):
    """
    Inspect a local dataset zip file and return stats.
    Supports both Detection/Segmentation and Classification formats.
    """
    if not zip_path: return {"status": "Error", "message": "No path provided"}
    
    path_obj = Path(zip_path)
    if not path_obj.exists():
            return {"status": "Error", "message": f"Path not found: {zip_path}"}
            
    # If it's already a directory, inspect it directly
    if path_obj.is_dir():
        temp_inspect_dir = path_obj
        is_temp = False
    else:
        # Extract to temp inspect folder
        temp_inspect_dir = Path(datasets_dir) / f"temp_inspect_{int(time.time())}_{random.randint(1000,9999)}"
        temp_inspect_dir.mkdir(parents=True, exist_ok=True)
        is_temp = True
        
        try:
            # Use smart extraction that auto-flattens nested structures
            extract_and_flatten_zip(zip_path, temp_inspect_dir)
        except Exception as e:
            if is_temp and temp_inspect_dir.exists(): shutil.rmtree(temp_inspect_dir)
            return {"status": "Error", "message": f"Extraction failed: {e}"}
            
    try:
        # Detect format type
        is_classification = (temp_inspect_dir / "train").exists() and (temp_inspect_dir / "train").is_dir()
        
        # Gather Stats
        stats = {
            "status": "Ready",
            "zip_path": str(zip_path),
            "images": {},
            "labels": {},
            "classes": 0,
            "class_names": []
        }
        
        if is_classification:
            # Classification format: train/dog/, val/dog/, test/dog/
            for subset in ["train", "val", "test"]:
                subset_dir = temp_inspect_dir / subset
                if subset_dir.exists():
                    # Count images across all class folders
                    img_count = 0
                    class_folders = [d for d in subset_dir.iterdir() if d.is_dir()]
                    
                    for class_folder in class_folders:
                        img_count += len([f for f in class_folder.iterdir() 
                                        if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                    
                    # Use capitalized names for consistency with UI
                    subset_key = subset.capitalize() if subset != "val" else "Validation"
                    stats["images"][subset_key] = img_count
                    stats["labels"][subset_key] = 0  # Classification doesn't have separate label files
                    
                    # Get class names from first subset that exists
                    if not stats["class_names"] and class_folders:
                        stats["class_names"] = sorted([d.name for d in class_folders])
                        stats["classes"] = len(stats["class_names"])
        else:
            # Detection/Segmentation format: images/Train/, labels/Train/
            for subset in ["Train", "Validation", "Test"]:
                img_dir = temp_inspect_dir / "images" / subset
                lbl_dir = temp_inspect_dir / "labels" / subset
                stats["images"][subset] = len(list(img_dir.iterdir())) if img_dir.exists() else 0
                stats["labels"][subset] = len(list(lbl_dir.iterdir())) if lbl_dir.exists() else 0
                
            yaml_path = temp_inspect_dir / "data.yaml"
            if yaml_path.exists():
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    stats["classes"] = data.get('nc', 0)
                    stats["class_names"] = data.get('names', [])
        
        # Apply robust sanitization before returning
        return _sanitize_stats_for_json(stats)
        
    except Exception as e:
        print(f"Error inspecting zip: {e}")
        return {"status": "Error", "message": str(e)}
    finally:
        if 'is_temp' in locals() and is_temp and temp_inspect_dir.exists(): 
            shutil.rmtree(temp_inspect_dir)
