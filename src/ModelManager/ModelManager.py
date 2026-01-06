import shutil
import sqlite3
import json
import zipfile
import time
from pathlib import Path

from cvat_sdk import make_client
from cvat_sdk.api_client import Configuration, ApiClient, models
import yaml
from DBmanager import DBManager

from config import CVAT_HOST_IP, CVAT_HOST_PORT, CVAT_USER, CVAT_PASSWORD
CVAT_HOST = CVAT_HOST_IP + ":" + CVAT_HOST_PORT


class ModelManager:
    def __init__ (self, db_path: str = 'model_manager.db'):
        """Initialize the ModelManager with a database path."""
        self.db_manager = DBManager(db_path)
        # self.cvat_client = make_client(
        #         host=CVAT_HOST_IP + ":" + CVAT_HOST_PORT,
        #         credentials=(CVAT_USER, CVAT_PASSWORD),
        # )
        self.cvat_client = ApiClient(Configuration(host=CVAT_HOST,
                                                   username=CVAT_USER,
                                                   password=CVAT_PASSWORD))

    def get_cvat_projects(self):
        """Fetch and return a list of CVAT projects."""
        with self.cvat_client as client:
            # First, get the organization ID for "PixeVision"
            org_id = None
            try:
                orgs, _ = client.organizations_api.list()
                for org in orgs.results:
                    if org.slug == "PixeVision" or org.name == "PixeVision":
                        org_id = org.id
                        print(f"Found organization 'PixeVision' with ID: {org_id}")
                        break
                if not org_id:
                    print("⚠️ Warning: Organization 'PixeVision' not found. Showing all projects.")
            except Exception as e:
                print(f"⚠️ Warning: Could not fetch organizations: {e}")
            
            # Fetch projects, filtered by organization if found
            if org_id:
                projects, _ = client.projects_api.list(org_id=org_id,page_size=100)
            else:
                projects, _ = client.projects_api.list()
            
            # Format as [(Name (ID: X), X)] for Gradio dropdown
            return [(f"{p.name} (ID: {p.id})", p.id) for p in projects.results]
      
    def download_and_format_project(self, project_id, format_name="Ultralytics YOLO Detection 1.0"):
        """
        Helper method to download and format a project to a specific location.
        Returns (SuccessBool, Message)
        """
                
        # --- PHASE 1: DOWNLOAD ---
        zip_path = ''
        project_name = ''
        try:
            print(f"Connecting to CVAT to download Project {project_id}...")
            with make_client(CVAT_HOST, credentials=(CVAT_USER, CVAT_PASSWORD)) as client:
                project = client.projects.retrieve(int(project_id))
                project_name = f"{project.name}_{project_id}"
                zip_path = Path('dataset') / f"{project_name}.zip"
                zip_path.parent.mkdir(parents=True, exist_ok=True)
                if zip_path.exists(): zip_path.unlink() 
                
                project.export_dataset(
                    format_name=format_name,
                    filename=str(zip_path),
                    include_images=True
                )
        except Exception as e:
            return False, f"Error downloading: {str(e)}"
        
        if not zip_path or not zip_path.exists():
            return False, "Error: Downloaded zip file not found."


        # --- PHASE 2: FORMAT ---
        dataset_dir = Path('dataset') / f"{project_name}"
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            
            # Detect format type
            is_classification = "Classification" in format_name
            if is_classification:
                # --- CLASSIFICATION FORMAT ---
                # CVAT exports may have capitalized folders: Train/, Validation/, Test/
                # But YOLO classification expects lowercase: train/, val/, test/
                if (dataset_dir/'Train').exists():
                    (dataset_dir/'Train').rename(dataset_dir/'train')
                if (dataset_dir/'Validation').exists():
                    (dataset_dir/'Validation').rename(dataset_dir/'val')
                if (dataset_dir/'Test').exists():
                    (dataset_dir/'Test').rename(dataset_dir/'test')
            
            else:
                # --- DETECTION/SEGMENTATION FORMAT ---
                # CVAT exports are already split: images/Train/, images/Validation/, images/Test/
                # Just copy the structure and update data.yaml paths
                if not (dataset_dir / "images").exists():
                    return False, "Error: 'images' directory not found."
                
                # Changing the txt files to point to correct image directories
                for split_name, txt_name in [("Train", "Train.txt"), ("Validation", "Validation.txt"), ("Test", "Test.txt")]:
                    txt_dir = dataset_dir / txt_name
                    if txt_dir.exists():
                        with open(txt_dir, "r", encoding="utf-8") as file:
                            content = file.read()

                        content = content.replace('data', txt_dir.as_posix())

                        with open(txt_dir, "w", encoding="utf-8") as file:
                            file.write(content)
                    
                        print(f"Fixed {split_name}")
                    else:
                        print(f"Warning: {split_name} directory not found, skipping {txt_name}")
                
                # Update data.yaml paths
                with open(dataset_dir/'data.yaml', "r") as f:
                    data = yaml.safe_load(f)
                for old_key, new_key in {"Test": "test", "Train": "train", "Validation": "val"}.items():
                    if old_key in data: data[new_key] = data.pop(old_key)
                with open(dataset_dir/'data.yaml', "w") as f:
                    yaml.dump(data, f, sort_keys=False)
            
            self.db_manager.create_dataset(
                cvat_project_id=project_id,
                name=project_name,
                format_type=format_name,
                storage_path=str(dataset_dir)
            )

            if zip_path.exists(): zip_path.unlink()
            return True, Path(dataset_dir)
        
        except Exception as e:
            return False, f"Format Error: {str(e)}"

    def train_model(self, project_id, model_name, epochs, imgsz=640, manual_aug=False, cvat_project_id=None, db_model_name="", db_model_version="", format_name="Ultralytics YOLO Detection 1.0", **kwargs):
        """Placeholder for model training logic."""

        # Check if dataset exists in DB, use it if exists
        dataset = self.db_manager.get_dataset_by_cvat_id(project_id)
        if dataset is None:
            success, dataset_path = self.download_and_format_project(project_id, format_name=format_name)
        else:
            dataset_path = dataset['storage_path']
        print(f"Using dataset at: {dataset_path}")

        # Extract augmentation parameters from kwargs
        aug_params = {}
        if manual_aug:
            # list of known augmentation args
            known_args = [
                'hsv_h', 'hsv_s', 'hsv_v', 'bgr', 
                'degrees', 'translate', 'scale', 'shear', 'perspective', 'flipud', 'fliplr',
                'mosaic', 'mixup', 'cutmix', 'copy_paste',
                'erasing'
            ]
            for key, value in kwargs.items():
                if key in known_args:
                    aug_params[key] = value
        
        




if __name__ == "__main__":
    manager = ModelManager()
    print(len(manager.get_cvat_projects()))
    download_result = manager.download_and_format_project(107, format_name="Ultralytics YOLO Detection 1.0")
    print(download_result)

    dataset_registry = DBManager('database.db')
    dataset = dataset_registry.get_dataset_by_cvat_id(107)
    dataset_path = dataset['storage_path']
    print(dataset_path)

    print(dataset)
    dataset_list = dataset_registry.list_datasets()
    print(dataset_list)

