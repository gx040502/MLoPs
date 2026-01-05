import os
import time
import json
import yaml
import zipfile
import shutil
from pathlib import Path
from http import HTTPStatus
from urllib.parse import parse_qsl, urlparse

from tqdm import tqdm
from ultralytics import settings

from cvat_sdk.api_client import Configuration, ApiClient, models

def extract_and_flatten_zip(zip_path, extract_to):
    """
    Intelligently extract zip and flatten if it contains only one root directory.
    
    This fixes the common issue where users zip a folder (abc.zip contains abc/)
    instead of zipping the folder contents directly.
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
    """
    
    # First, extract to a temporary location
    temp_extract = Path(extract_to).parent / f"temp_{Path(extract_to).name}"
    temp_extract.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract)
        
        # Check if there's only one root directory
        root_items = list(temp_extract.iterdir())
        
        if len(root_items) == 1 and root_items[0].is_dir():
            # Single root directory - flatten by moving its contents up
            single_root = root_items[0]
            print(f"🔄 Flattening nested structure: {single_root.name}/")
            
            # Move contents from nested folder to final destination
            if Path(extract_to).exists():
                shutil.rmtree(extract_to)
            shutil.move(str(single_root), str(extract_to))
        else:
            # Multiple root items or root is not a directory - use as is
            if Path(extract_to).exists():
                shutil.rmtree(extract_to)
            shutil.move(str(temp_extract), str(extract_to))
            
    finally:
        # Clean up temp directory if it still exists
        if temp_extract.exists():
            shutil.rmtree(temp_extract)

def upload_dataset_by_zip(zip_file_path, datasets_dir, config, save_config_callback):
    """
    Upload and extract a raw dataset from a zip file, then save it to the datasets directory in settings.json.
    
    Args:
        zip_file_path: Path to the zip file to upload
        datasets_dir: Directory where datasets are stored
        config: Configuration dictionary to update
        save_config_callback: Function to call to save config changes
        
    Returns:
        tuple: (success: bool, message: str, project_name: str or None)
    """
    
    if zip_file_path is None:
        return False, "No file was uploaded. Please upload a ZIP file.", None
    
    # Check if the uploaded file is a ZIP file
    if not zipfile.is_zipfile(zip_file_path):
        return False, "The uploaded file is not a valid ZIP file. Please upload a .zip archive.", None
    
    try:
        project_name = zip_file_path.split('/')[-1].replace('.zip', '')
        output_dir = os.path.join(datasets_dir, project_name)
        
        # Use smart extraction that auto-flattens nested structures
        extract_and_flatten_zip(zip_file_path, output_dir)
        
        # Update the config with the new dataset
        if "datasets" not in config:
            config["datasets"] = []
        config["datasets"].append({"name": project_name, "path": output_dir})
        save_config_callback()

        return True, f"Successfully unzipped the file to '{output_dir}'.", project_name
    except Exception as e:
        return False, f"An error occurred while unzipping the file: {e}", None
