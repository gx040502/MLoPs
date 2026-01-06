import shutil
import sqlite3
import json
import zipfile
import time
from pathlib import Path

from cvat_sdk import make_client
from cvat_sdk.api_client import Configuration, ApiClient, models
import yaml
from ultralytics import YOLO
import numpy as np

from DBmanager import DBManager
from config import CVAT_HOST_IP, CVAT_HOST_PORT, CVAT_USER, CVAT_PASSWORD
CVAT_HOST = CVAT_HOST_IP + ":" + CVAT_HOST_PORT


class ModelManager:
    def __init__ (self, db_path: str = 'database.db'):
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
                zip_path = Path('datasets') / f"{project_name}.zip"
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
        dataset_dir = Path('datasets') / f"{project_name}"
        
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

    def get_models(self, cvat_project_id=None, name=None):
        """Retrieve a list of models from the database."""
        return self.db_manager.list_models(cvat_project_id=cvat_project_id, name=name)
    
    def get_datasets(self, cvat_project_id=None, name=None):
        """Retrieve a list of datasets from the database."""
        return self.db_manager.list_datasets(cvat_project_id=cvat_project_id, name=name)

    def load_model(self, model_id: int) -> YOLO:
        """Load a YOLO model from the database by model ID."""
        model_record = self.db_manager.get_model(model_id)
        if not model_record:
            print(f"Model with ID {model_id} not found in database.")
            return None
        else:
            model_path = model_record['storage_path']
            return YOLO(model_path)
    
    def inference_image(self, image: np.ndarray, model: YOLO, conf: float, iou: float, verbose: bool = False) -> list:
        try:
            if model.task == "classify":
                results = model.predict(image, device=0, verbose=verbose)

                #Format classification results
                result = results[0]
                top5_indices = result.probs.top5
                top5_conf = result.probs.top5conf.tolist()
                
                prediction_details = {
                    "task": "classification",
                    "top_predictions": []
                }
                
                for idx, conf in zip(top5_indices, top5_conf):
                    class_name = result.names[idx]
                    prediction_details["top_predictions"].append({
                        "class": class_name,
                        "confidence": float(conf)
                    })
                
                # Return annotated image (classification doesn't change image much, so maybe just original or top1 text)
                # But YOLO plot() for classify just returns the image usually
                start_time = time.time()
                annotated_img = result.plot()
                postprocess_time = (time.time() - start_time) * 1000
                if verbose: print(f"Prediction done. Time: {postprocess_time:.2f}ms")
                
                return annotated_img, json.dumps(prediction_details, indent=2)
            else:
                # Detection/Segmentation
                if verbose: print(f"Running detection/segmentation prediction on: {Path(model.model_name).name}")
                start_time = time.time()
                params = {"conf": conf, "iou": iou, "device": 0, "verbose": False}
                results = model.predict(image, **params)
                
                res = results[0]
                annotated_img = res.plot()
                postprocess_time = (time.time() - start_time) * 1000
                if verbose: print(f"Prediction done. Time: {postprocess_time:.2f}ms")
                
                # Format detection results
                detections = []
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    class_name = res.names[cls_id]
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    detections.append({
                        "class": class_name,
                        "confidence": conf,
                        "bbox": xyxy
                    })
                
                return annotated_img, json.dumps(detections, indent=2)
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            return None, None

    def inference_video(self, video_path: str, model: YOLO, conf: float, iou: float, verbose: bool = False):
        """Perform inference on a video file using the specified model."""
        try:
            if verbose: print(f"Running video inference on: {Path(model.model_name).name}")
            start_time = time.time()
            params = {"conf": conf, "iou": iou, "device": 0, "verbose": False}
            
            # Define input and output paths (adjust as necessary)
            input_path = video_path
            output_path = 'output.mp4'
            # Convert AVI to MP4
            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' or 'XVID' codec
            out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(frame, **params)
                out.write(results[0].plot())
            cap.release()
            out.release()
            print(f"Saved MP4 video to {output_path}")

            postprocess_time = (time.time() - start_time) * 1000
            if verbose: print(f"Video prediction done. Time: {postprocess_time:.2f}ms")
            return results
        except Exception as e:
            print(f"Error during video inference: {str(e)}")
            return None
        
        

    
if __name__ == "__main__":
    manager = ModelManager()
    print(len(manager.get_cvat_projects()))

    model = YOLO("/home/cy/projects/SEEAI/src/ModelManager/yolo11n.pt")

    # model_id = manager.db_manager.register_model(
    #     cvat_project_id=107,
    #     name="test_model",
    #     version="v1.0",
    #     model= model
    # )

    # download_result = manager.download_and_format_project(107, format_name="Ultralytics YOLO Detection 1.0")
    # print(download_result)

    # dataset_registry = DBManager('database.db')
    # dataset = dataset_registry.get_dataset_by_cvat_id(107)
    # dataset_path = dataset['storage_path']
    # print(dataset_path)

    # print(dataset)
    # dataset_list = dataset_registry.list_datasets()
    # print(dataset_list)
    import cv2
    img = cv2.imread('/home/cy/projects/SEEAI/examples/weng_yeng.png')
    video_path = '/home/cy/projects/SEEAI/examples/lentera_site.mp4'
    result = manager.inference_video(video_path, model, conf=0.25, iou=0.45)

    print(manager.get_models(cvat_project_id=107))

