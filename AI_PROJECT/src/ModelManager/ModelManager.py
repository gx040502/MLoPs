import shutil
import sqlite3
import json
import zipfile
import time
from pathlib import Path
from cvat_sdk import make_client
from cvat_sdk.api_client import Configuration, ApiClient, models
import yaml
from .DBmanager import DBManager
from . import model_trainer
from AI_PROJECT.config.config import CVAT_HOST_IP, CVAT_HOST_PORT, CVAT_USER, CVAT_PASSWORD
from ultralytics import YOLO
import numpy as np
import gradio as gr
import os
import cv2
import tempfile

CVAT_HOST = CVAT_HOST_IP + ":" + CVAT_HOST_PORT

#hi
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
      
    def download_and_format_project(self, project_id, format_name="Ultralytics YOLO Detection 1.0", custom_name=None):
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
                
                if custom_name and custom_name.strip():
                    project_name = custom_name.strip()
                    if project_name.lower().endswith('.zip'):
                        project_name = project_name[:-4]
                else:
                    project_name = f"{project.name}_{project_id}"
                
                
                # Base directory for ModelManager datasets (../../data)
                base_dir = Path(__file__).resolve().parents[2].joinpath('data')
                
                # Zip file goes to project root 'dataset' folder as requested
                zip_path = base_dir / 'datasets' / f"{project_name}.zip"
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
        dataset_dir = base_dir / 'datasets' / f"{project_name}"
        
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
                # We need to ensure absolute paths are used in data.yaml and text files are pointing to correct locations
                
                abs_dataset_dir = dataset_dir.resolve()
                
                if not (dataset_dir / "images").exists():
                    return False, "Error: 'images' directory not found."
                
                # 1. Update data.yaml with absolute paths
                yaml_path = dataset_dir / "data.yaml"
                if yaml_path.exists():
                    with open(yaml_path, 'r') as f:
                        yaml_content = f.read()
                    
                    # Remove old path lines (both lowercase and capitalized versions, and validation variants)
                    filtered_lines = [l for l in yaml_content.splitlines() if not (
                        l.strip().lower().startswith('train:') or 
                        l.strip().lower().startswith('val:') or 
                        l.strip().lower().startswith('validation:') or 
                        l.strip().lower().startswith('test:')
                    )]
                    
                    # Build header lines only for splits that exist
                    header_lines = []
                    for split_name in ["Train.txt", "Validation.txt", "Test.txt"]:
                        split_path = abs_dataset_dir / split_name
                        if split_path.exists():
                            # Use lowercase keys for YOLO (train, val, test)
                            yaml_key = split_name.replace('.txt', '').lower()
                            if yaml_key == 'validation':
                                yaml_key = 'val'
                            header_lines.append(f"{yaml_key}: {split_path}")
                        else:
                            print(f"Info: {split_name} not found, skipping from data.yaml")
                    
                    final_yaml_content = "\n".join(header_lines) + "\n" + "\n".join(filtered_lines)
                    
                    with open(yaml_path, 'w') as f:
                        f.write(final_yaml_content)
                
                # 2. Update Train.txt, Validation.txt, Test.txt
                # The goal is to make sure lines in txt files are absolute paths to images
                for txt_name in ["Train.txt", "Validation.txt", "Test.txt"]:
                    txt_path = dataset_dir / txt_name
                    if txt_path.exists():
                        with open(txt_path, 'r') as f:
                            lines = f.readlines()
                        
                        new_lines = []
                        for line in lines:
                            line = line.strip()
                            if not line: continue
                            
                            # CVAT export usually puts "./images/Train/img.jpg" or "data/images/Train/img.jpg"
                            # We want absolute path
                            if line.startswith("./"):
                                # If it starts with ./, resolve it relative to dataset root
                                new_lines.append(str(abs_dataset_dir / line[2:]) + "\n")
                            elif line.startswith("data/"):
                                # "data" was likely the placeholder relative path
                                new_lines.append(str(abs_dataset_dir / line.replace("data/", "", 1)) + "\n")
                            else:
                                # Fallback or already absolute?
                                # If the path is not absolute, assume it's relative to dataset root
                                # We skip the expensive .exists() check to speed up processing for large datasets
                                path_obj = Path(line)
                                if not path_obj.is_absolute():
                                     new_lines.append(str(abs_dataset_dir / line) + "\n")
                                else:
                                     new_lines.append(line + "\n")
                        
                        with open(txt_path, 'w') as f:
                            f.writelines(new_lines)
                        print(f"Fixed {txt_name}")
                    else:
                        print(f"Warning: {txt_name} not found")
            
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
        """
        Orchestrates the training process:
        1. Ensure dataset is available (download/format if needed)
        2. Call train.py
        3. Register the trained model in database
        """
        
        # 1. Get Dataset
        dataset = self.db_manager.get_dataset_by_cvat_id(project_id)
        if dataset is None:
            print(f"Dataset for project {project_id} not found locally. Downloading...")
            success, result_or_msg = self.download_and_format_project(project_id, format_name=format_name)
            if not success:
                return f"❌ Error preparing dataset: {result_or_msg}"
            dataset_path = str(result_or_msg)
        else:
            dataset_path = dataset['storage_path']
            print(f"Using existing dataset at: {dataset_path}")

        # Extract augmentation parameters from kwargs
        aug_params = {}
        if manual_aug:
            known_args = [
                'hsv_h', 'hsv_s', 'hsv_v', 'bgr', 
                'degrees', 'translate', 'scale', 'shear', 'perspective', 'flipud', 'fliplr',
                'mosaic', 'mixup', 'cutmix', 'copy_paste',
                'erasing'
            ]
            for key, value in kwargs.items():
                if key in known_args:
                    aug_params[key] = value

        # 2. Call train.py
        try:
            # Import train module (assumed to be in path or project root)
            import sys
            # Attempt to add project root to path if not present (../../)
            project_root = str(Path(__file__).resolve().parents[2])
            if project_root not in sys.path:
                sys.path.append(project_root)
            
            project_name = Path(dataset_path).name
            
            # Construct absolute model path
            sub_dir = "detection"
            if format_name and ("Segmentation" in format_name):
                 sub_dir = "segmentation"
            elif format_name and ("Classification" in format_name):
                 sub_dir = "classification"

            # Assuming standard model naming convention in models/pre_trained/
            base_dir = Path(__file__).resolve().parents[2].joinpath('data')
            model_file = base_dir / "models/pre_trained" / sub_dir / model_name
            
            print(f"Starting training with model: {model_file}")

            success, msg = model_trainer.run_training(
                project_name=project_name,
                model_path=str(model_file),
                epochs=int(epochs),
                imgsz=int(imgsz),
                manual_aug=manual_aug,
                aug_params=aug_params,
                format_name=format_name
            )

            if success:
                # 3. Register model in database
                try:
                    if cvat_project_id:
                        # Determine final model name and version
                        final_model_name = db_model_name.strip() if db_model_name.strip() else f"Project_{project_name}"
                        
                        # Prepare project ID
                        project_id_int = int(str(cvat_project_id).split(':')[0].strip()) if isinstance(cvat_project_id, str) else cvat_project_id
                        
                        if db_model_version.strip():
                            final_version = db_model_version.strip()
                        else:
                            # Auto-increment version
                            models_list = self.db_manager.list_models(cvat_project_id=project_id_int)
                            final_version = f"v{len(models_list) + 1}"
                        
                        # Find the trained model weights
                        # We use the logic from app.py to find the latest run directory in models/train/{project_name}
                        # because model_trainer.run_training returns a message string, not a clean path.
                        
                        base_dir = Path(__file__).resolve().parents[2].joinpath('data')
                        project_train_dir = base_dir / "models/train" / project_name
                        best_model_path = None
                        
                        if project_train_dir.exists():
                            # Find the most recent run directory (has weights/best.pt)
                            # Sort by mtime reverse to check newest first
                            sorted_runs = sorted(project_train_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
                            for run_dir in sorted_runs:
                                if run_dir.is_dir():
                                    potential_path = run_dir / "weights" / "best.pt"
                                    if potential_path.exists():
                                        best_model_path = potential_path
                                        break
                        
                        if best_model_path and best_model_path.exists():
                            model_id = self.db_manager.register_model(
                                cvat_project_id=project_id_int,
                                name=final_model_name,
                                version=final_version,
                                model=str(best_model_path)
                            )
                            print(f"✅ Model registered in database with ID: {model_id}, Name: {final_model_name} {final_version}")
                            msg += f"\n\n📊 Model registered: {final_model_name} {final_version} (ID: {model_id})"
                        else:
                            print(f"⚠️ Best model not found at {best_model_path}, skipping registration.")
                            
                except Exception as reg_err:
                    print(f"⚠️ Failed to register model in database: {reg_err}")
                    import traceback
                    traceback.print_exc()

                return True, msg, best_model_path
            else:
                return False, msg, None

        except Exception as e:
            return False, f"Error invoking training: {str(e)}", None
    def get_trained_projects(self):
        return self.db_manager.get_unique_projects()
    
    def get_model(self, model_id):
        """Retrieve a model from the database."""
        return self.db_manager.get_model(model_id)

    def get_models(self, cvat_project_id=None, name=None):
        """Retrieve a list of models from the database for a project."""
        return self.db_manager.list_models(cvat_project_id=cvat_project_id, name=name)

    def get_pretrained_models(self):
        """Retrieve a list of pretrained models from the database."""
        return self.db_manager.list_pretrained_models()

    def get_model_dataset(self,project_id):
        """Retrieve a dataset from the database."""
        return self.db_manager.get_dataset_by_cvat_id(project_id)

    def get_model_datasets(self, cvat_project_id=None, name=None):
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
            print(f"Loading model from path: {model_path}")
            return YOLO(model_path)
    
    def get_model_trained_args(self, model_path):
        """
        Load training arguments from args.yaml in the model's training directory.
        
        Args:
            model_path: Path to the model file (e.g., .../weights/best.pt)
            
        Returns:
            dict: Training arguments, or None if not found
        """
        # Convert to Path object if it's a string
        if isinstance(model_path, str):
            model_path = Path(model_path)
        
        # Navigate from best.pt -> weights -> training_run_dir
        args_dir = model_path.parent.parent
        args_file = args_dir / "args.yaml"
        
        if args_file.exists():
            try:
                with open(args_file, 'r') as f:
                    args = yaml.safe_load(f)
                return args
            except Exception as e:
                print(f"Error loading args.yaml: {e}")
                return None
        else:
            print(f"args.yaml not found at {args_file}")
            return None

    def get_test_images(self, project_id):
        """
        Returns a list of image paths from Test AND Validation folders combined
        Supports both Detection/Segmentation (images/Test, images/Validation) and Classification (test/, val/) formats
        Combines images from: Test + test + Validation + val
        """

        dataset=self.get_model_dataset(project_id)
        dataset_path=str(dataset["storage_path"])
     
        # Define all possible directories to collect from
        possible_dirs = [
            Path(dataset_path) / "images" / "Test",        # Detection/Segmentation Test
            Path(dataset_path) / "test",                    # Classification test
            Path(dataset_path) / "images" / "Validation",   # Detection/Segmentation Validation
            Path(dataset_path) / "val",                     # Classification val
        ]
        
        # Collect images from ALL available directories
        all_images = []
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        folders_used = []
        
        for directory in possible_dirs:
            if directory.exists():
                dir_images = []
                
                # Check based on directory type
                if directory.name in ["test", "val"]:
                    # Classification format: traverse class folders
                    for class_folder in directory.iterdir():
                        if class_folder.is_dir():
                            for img_path in class_folder.iterdir():
                                if img_path.is_file() and img_path.suffix.lower() in valid_exts:
                                    dir_images.append(str(img_path.resolve()))
                else:
                    # Detection/Segmentation format: images directly in folder
                    for img_path in directory.iterdir():
                        if img_path.is_file() and img_path.suffix.lower() in valid_exts:
                            dir_images.append(str(img_path.resolve()))
                
                # Add to combined list if we found images
                if dir_images:
                    all_images.extend(dir_images)
                    folders_used.append(directory.name)
        
        # Print which folders were used
        if folders_used:
            print(f"📂 Collecting test images from: {', '.join(folders_used)}")
        else:
            print("⚠️ No test or validation images found")
            return []
        
        # Remove duplicates and sort
        unique_images = sorted(set(all_images))
        
        # Limit to avoid overloading UI if too many
        return unique_images[:50] 

    
    def create_pretrained_model(self, model_name: str, model_path: str):
        """
        Save an uploaded model to the models/uploaded directory and register it in the database.
        """
        try:
            # 1. Prepare directory
            base_dir = Path(__file__).resolve().parents[2].joinpath('data')
            upload_dir = base_dir / "models/uploaded"
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # 2. Copy file to destination
            # Ensure safe filename
            safe_name = "".join([c for c in model_name if c.isalnum() or c in (' ', '_', '-')]).strip().replace(" ", "_")
            dest_path = upload_dir / f"{safe_name}.pt"
            
            shutil.copy2(model_path, dest_path)
            
            # 3. Register in DB
            # Use -1 for cvat_project_id to indicate it's a general/uploaded model
            model_id = self.db_manager.register_model(
                cvat_project_id=-1,
                name=model_name,
                version="1",
                model=str(dest_path)
            )
            
            return True, f"Model uploaded and registered successfully with ID: {model_id}"
            
        except Exception as e:
            return False, f"Error creating pretrained model: {str(e)}"
    
    def delete_model(self, model_id: int):
        """
        Delete a model from the database and the filesystem.
        """
        try:
            # 1. Get model info to find path
            model_record = self.db_manager.get_model(model_id)
            if not model_record:
                return False, "Model not found in database."
            
            model_path = Path(model_record['storage_path'])
            
            # 2. Delete from Database
            db_success = self.db_manager.delete_model(model_id)
            if not db_success:
                return False, "Failed to delete model from database."
            
            # 3. Delete from Filesystem
            # Only delete if it is in the uploaded directory to avoid deleting pre-installed models??
            # Or just delete if it exists. Let's rely on the fact that these are 'pretrained/uploaded' models.
            if model_path.exists():
                try:
                    model_path.unlink()
                    return True, "Model deleted from database and filesystem."
                except Exception as e:
                    return True, f"Model deleted from DB, but failed away file: {e}"
            else:
                 return True, "Model deleted from DB (file was missing)."
                 
        except Exception as e:
            return False, f"Error deleting model: {str(e)}"
    
    def inference_image(self, image: np.ndarray, model: YOLO, conf: float, iou: float, verbose: bool = False) -> list:
        try:
            print(model.task)
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
                annotated_img = result.plot()  # Returns BGR
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)  # Convert to RGB for Gradio
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
                annotated_img = res.plot()  # Returns BGR
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)  # Convert to RGB for Gradio
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

    def inference_video(self, video_path: str, model: YOLO, conf: float, iou: float, output_extension: str = '.webm', verbose: bool = False):
        """Perform inference on a video file using the specified model."""
        try:
            if verbose: print(f"Running video inference on: {getattr(model, 'model_name', 'model')}")
            start_time = time.time()
            params = {"conf": conf, "iou": iou, "device": 0, "verbose": False}
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, json.dumps({"error": "Failed to open video"}, indent=2)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Ensure extension has dot
            if not output_extension.startswith('.'):
                output_extension = f".{output_extension}"
            
            # Codec mapping
            codecs = {
                '.mp4': 'mp4v',
                '.avi': 'XVID',
                '.mov': 'mp4v',
                '.mkv': 'VP09',
                '.webm': 'vp09'
            }
            
            # Default to webm/vp09 if unknown
            # For mp4, 'avc1' or 'h264' is better if available but 'mp4v' is safer for opencv default
            # For webm, try vp09, fallback to VP80
            
            ext = output_extension.lower()
            codec_str = codecs.get(ext, 'vp09')
            
            # Create temp output path
            output_path = Path(tempfile.mkdtemp()) / f"prediction{ext}"
            
            fourcc = cv2.VideoWriter_fourcc(*codec_str)
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            # Check if writer opened successfully
            if not out.isOpened():
                print(f"Codec {codec_str} failed for {ext}, trying fallbacks...")
                if ext == '.webm':
                     fourcc = cv2.VideoWriter_fourcc(*'VP80')
                elif ext == '.mp4':
                     fourcc = cv2.VideoWriter_fourcc(*'avc1')
                else:
                     fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Generic fallback
                
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            frame_count = 0
            total_detections = 0

            print(f"Processing video: {total_frames} frames at {fps} FPS")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(frame, **params)
                
                # Count detections
                if hasattr(results[0], 'boxes'):
                    total_detections += len(results[0].boxes)
                
                out.write(results[0].plot())
                frame_count += 1
                
            cap.release()
            out.release()
            print(f"Saved video to {output_path}")

            postprocess_time = (time.time() - start_time) * 1000
            if verbose: print(f"Video prediction done. Time: {postprocess_time:.2f}ms")
            
            details = {
                "task": "video_prediction",
                "frames_processed": frame_count,
                "total_detections": total_detections,
                "fps": fps,
                "resolution": f"{width}x{height}"
            }
            
            return output_path, json.dumps(details, indent=2)
            
        except Exception as e:
            print(f"Error during video inference: {str(e)}")
            return None, json.dumps({"error": str(e)}, indent=2)
    
    def display_model_details(self, model_id, model_trained_args=None):
        """Display selected model details with rich HTML formatting"""
        if not model_id:
            return "<p>Select a model to view details</p>", gr.update(visible=False)
        
        try:
            # Get model path from config
            own_models = self.get_models()
            selected_model = next((m for m in own_models if m.get("id") == model_id), None)
            
            if not selected_model:
                return "<p>Model not found in configuration</p>", gr.update(visible=False)
            
            model_name = selected_model["name"]
            model_path = selected_model["storage_path"]
            
            if not model_path or not os.path.exists(model_path):
                return "<p>Model file not found</p>", gr.update(visible=False)
                
            # Load model and get info
            model = YOLO(model_path)
            registry = DBManager('database.db')
            model_info = registry.get_pt_model_info(model)
            

            # Format labels - each on new line
            labels = model_info['labels']
            label_str = '\n'.join([f"{k}: {v}" for k, v in labels.items()]) if isinstance(labels, dict) else str(labels)
            
            # Calculate score percentage
            score_pct = model_info.get('primary_score', 0) * 100
            score_color = "#34d399" if score_pct > 80 else "#fbbf24" if score_pct > 50 else "#f87171"
            
            # Format trained date to human-readable
            trained_at_raw = model_info.get('trained_at', 'N/A')
            if trained_at_raw != 'N/A':
                try:
                    # Parse ISO format datetime
                    dt = datetime.fromisoformat(trained_at_raw.replace('Z', '+00:00'))
                    trained_at_formatted = dt.strftime('%b %d, %Y %I:%M %p')
                except:
                    trained_at_formatted = trained_at_raw
            else:
                trained_at_formatted = 'N/A'
            
            # Format metrics
            metrics = model_info.get('metrics', {})
            
            # Format training args if provided
            training_args_html = ""
            if model_trained_args:
                # Select key training parameters to display
                key_args = ['epochs', 'batch', 'imgsz', 'optimizer', 'lr0', 'lrf', 
                           'momentum', 'weight_decay', 'warmup_epochs', 'hsv_h', 
                           'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 
                           'fliplr', 'mosaic', 'mixup']
                
                args_rows = []
                for key in key_args:
                    if key in model_trained_args:
                        value = model_trained_args[key]
                        # Format the value nicely
                        if isinstance(value, float):
                            value_str = f"{value:.4f}"
                        else:
                            value_str = str(value)
                        args_rows.append(f'<div class="data-row"><span class="data-label">{key}</span><span class="data-value">{value_str}</span></div>')
                
                if args_rows:
                    training_args_html = f"""
                    <div class="info-card">
                        <div class="card-title">⚙️ Training Config</div>
                        {''.join(args_rows[:12])}
                    </div>
                    """
            
            html = f"""
            <style>
                .model-dashboard {{
                    font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
                    color: #e5e7eb;
                    max-width: 100%;
                }}
                .header-section {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 20px;
                    gap: 12px;
                }}
                .model-title {{ 
                    font-size: 1.5rem; 
                    font-weight: 700; 
                    margin: 0; 
                    color: #ffffff;
                }}
                .info-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: 16px;
                }}
                .info-card {{
                    background: #1f2937;
                    border: 1px solid #374151;
                    border-radius: 12px;
                    padding: 16px;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
                }}
                .card-title {{
                    font-size: 0.95rem; 
                    font-weight: 600; 
                    color: #9ca3af;
                    margin-bottom: 12px; 
                    text-transform: uppercase; 
                    letter-spacing: 0.05em;
                    border-bottom: 1px solid #374151;
                    padding-bottom: 8px;
                }}
                .data-row {{
                    display: flex; 
                    justify-content: space-between;
                    margin-bottom: 8px; 
                    font-size: 0.9rem;
                }}
                .data-label {{ color: #d1d5db; }}
                .data-value {{ font-weight: 500; color: #ffffff; text-align: right; }}
                .progress-bg {{
                    background: #374151;
                    height: 8px; 
                    width: 100%; 
                    border-radius: 4px; 
                    margin-top: 6px; 
                    overflow: hidden;
                }}
                .progress-fill {{ 
                    height: 100%; 
                    border-radius: 4px; 
                    transition: width 0.3s ease; 
                    box-shadow: 0 0 8px {score_color};
                }}
                .classes-box {{
                    background: #111827;
                    padding: 8px;
                    border: 1px solid #374151;
                    border-radius: 6px; 
                    font-size: 0.8rem; 
                    color: #9ca3af;
                    white-space: pre-line; 
                    max-height: 150px;
                    overflow-y: auto;
                }}
            </style>
            
            <div class="model-dashboard">
                <div class="header-section">
                    <h3 class="model-title">{model_name}</h3>
                </div>
                
                <div class="info-grid">
                    <div class="info-card">
                        <div class="card-title">🎯 Performance</div>
                        <div class="data-row">
                            <span class="data-label">Task</span>
                            <span class="data-value">{model_info['task'].title()}</span>
                        </div>
                        <div style="margin-bottom: 12px;">
                            <div class="data-row" style="margin-bottom:2px;">
                                <span class="data-label">{model_info['score_type']}</span>
                                <span class="data-value" style="color: {score_color}">{model_info['primary_score']:.4f}</span>
                            </div>
                            <div class="progress-bg">
                                <div class="progress-fill" style="width: {score_pct}%; background: {score_color};"></div>
                            </div>
                        </div>
                        <div class="data-row">
                            <span class="data-label">🏷️ Total Classes</span>
                            <span class="data-value">{len(labels)}</span>
                        </div>
                        <div class="classes-box" title="{label_str}">{label_str}</div>
                    </div>
                    
                    <div class="info-card">
                        <div class="card-title">📦 Storage & Hardware</div>
                        <div class="data-row">
                            <span class="data-label">File Size</span>
                            <span class="data-value">{model_info.get('model_size_mb', 0):.2f} MB</span>
                        </div>
                        <div class="data-row">
                            <span class="data-label">VRAM Usage</span>
                            <span class="data-value">{model_info.get('vram_gb', 0):.2f} GB</span>
                        </div>
                        <div class="data-row">
                            <span class="data-label">Trained Date</span>
                            <span class="data-value">{trained_at_formatted}</span>
                        </div>
                        <div style="margin-top:10px; font-size:0.75rem; color:#6b7280;">
                            Path: ...{model_path[-30:]}
                        </div>
                    </div>
                    
                    <div class="info-card">
                        <div class="card-title">📈 Key Metrics</div>
                        {''.join([
                            f'<div class="data-row"><span class="data-label">{k}</span><span class="data-value">{v if isinstance(v, str) else f"{v:.4f}"}</span></div>' 
                            for k, v in list(metrics.items())[:6]
                        ])}
                    </div>
                    
                    {training_args_html}
                </div>
            </div>
            """
            return html, gr.update(visible=True)
            
        except Exception as e:
            return f"<p>Error loading model details: {str(e)}</p>", gr.update(visible=False)
    
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

