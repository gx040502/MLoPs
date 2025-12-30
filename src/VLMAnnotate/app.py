import traceback
import json
import time
import os
import shutil
import zipfile
import random
import io
from pathlib import Path

import torch
import matplotlib
import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from cvat_sdk import make_client
from cvat_sdk.api_client import Configuration, ApiClient, models
from types import SimpleNamespace

from .utils import COCODatasetBuilder, GroundingDINODetector
from .utils import COCODatasetBuilder, GroundingDINODetector
from ultralytics import YOLO, SAM

class APP():

    def __init__(self, vlm_model: GroundingDINODetector=None):
        self.json_path = 'settings.json'
        try:
            with open(self.json_path, 'r') as config_file:
                self.config = json.load(config_file)
                self.datasets_dir = self.config.get("datasets_dir", ".gradio")
                self.html = self.create_dataset_html()
                self.selected_dataset = ''
                self.selected_dataset_1st_img_path = ""

            self.sam_model = SAM("sam2.1_b.pt")
            self.model = vlm_model
            # Setup Training directory path
            self.train_root_dir = Path("/home/intern/Gitlab/pipeline/1.Train")

        except (FileNotFoundError, json.JSONDecodeError):
            print("Could not load settings.json. Using default configuration.")
            self.config = {}
            self.datasets_dir = ".gradio"
            self.train_root_dir = Path("/home/intern/Gitlab/pipeline/1.Train")

    def select_dataset(self, name):
        
        success, dataset = self.get_dataset_by_name(name)
        if success:
            self.selected_dataset = name
            # Get first image in dataset directory for preview
            dataset_path = dataset.get("path", "")
            if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
                for file in os.listdir(dataset_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        self.selected_dataset_1st_img_path = os.path.join(dataset_path, file)
                        break
                else:
                    self.selected_dataset_1st_img_path = ''
            else:
                self.selected_dataset_1st_img_path = ''
            msg = f"Path: {dataset.get('path', 'N/A')}"
            return True, msg
        return False, f"Dataset '{name}' not found."

    # -------------------------------------------------------------------------
    #                         PREDICT MODEL LOGIC
    # -------------------------------------------------------------------------
     
    def extract_and_flatten_zip(self, zip_path, extract_to):
        """
        Intelligently extract zip and flatten if it contains only one root directory.
        
        This fixes the common issue where users zip a folder (abc.zip contains abc/)
        instead of zipping the folder contents directly.
        
        Args:
            zip_path: Path to zip file
            extract_to: Directory to extract to
        """
        import zipfile
        
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
    
    def list_trained_projects(self):
        """
        Scans the 1.Train directory for project folders.
        Returns a list of project names.
        """
        if not self.train_root_dir.exists():
            return []
        
        projects = []
        for item in self.train_root_dir.iterdir():
            if item.is_dir():
                projects.append(item.name)
        return sorted(projects)

    def list_trained_models(self, project_name):
        """
        Scans 1.Train/<project_name> for run directories that contain weights/best.pt.
        Returns a list of model names (run names).
        """
        if not project_name:
            return []
            
        project_dir = self.train_root_dir / project_name
        if not project_dir.exists():
            return []
            
        models_list = []
        for run_dir in project_dir.iterdir():
            if run_dir.is_dir():
                # Check for weights/best.pt
                best_pt = run_dir / "weights" / "best.pt"
                if best_pt.exists():
                    models_list.append(run_dir.name)
        
        return sorted(models_list)

    def get_test_images(self, project_name):
        """
        Returns a list of image paths from Dataset/{project_name}/images/Test or test
        Supports both Detection/Segmentation (images/Test) and Classification (test) formats
        """
        if not project_name: return []
        
        # Dataset assumes CWD is project root
        project_dir = Path("Dataset") / project_name
        
        # Try Detection/Segmentation format: images/Test
        test_dir = project_dir / "images" / "Test"
        
        # If not found, try Classification format: test/
        if not test_dir.exists():
            test_dir = project_dir / "test"
        
        # If still not found, return empty
        if not test_dir.exists():
            return []
        
        images = []
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # For Classification, test/ contains class subfolders (dog/, cat/, etc.)
        # Collect images from all subfolders
        if test_dir.name == "test":
            # Classification format: traverse class folders
            for class_folder in test_dir.iterdir():
                if class_folder.is_dir():
                    for img_path in class_folder.iterdir():
                        if img_path.is_file() and img_path.suffix.lower() in valid_exts:
                            images.append(str(img_path.resolve()))
        else:
            # Detection/Segmentation format: images directly in Test/
            for img_path in test_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in valid_exts:
                    images.append(str(img_path.resolve()))
                
        # Limit to avoid overloading UI if too many
        return sorted(images)[:50] 

    def get_model_plots(self, project_name, model_name):
        """
        Returns list of plot images from 1.Train/{project}/{model}/...
        Specific files: BoxF1_curve, BoxP_curve, BoxPR_curve, BoxR_curve,
        confusion_matrix_normalized, confusion_matrix, labels, results.
        """
        if not project_name or not model_name: return []
        
        model_dir = self.train_root_dir / project_name / model_name
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
                
        # Return list of (label, path) tuples for Gallery, or just paths?
        # Gradio Gallery accepts list of (path, label) tuples.
        return [(path, label) for label, path in plots]

    def get_model_details(self, project_name, model_name):
        """Returns details about the selected model."""
        if not project_name or not model_name:
            return ""
            
        model_path = self.train_root_dir / project_name / model_name / "weights" / "best.pt"
        if not model_path.exists():
            return "❌ Model file not found."
            
        return f"Path: {model_path}\nSize: {model_path.stat().st_size / (1024*1024):.2f} MB"

    def predict_with_model(self, project_name, model_name, image, conf_threshold, iou_threshold):
        """
        Runs YOLO inference on the image using the selected model.
        Supports both detection/segmentation and classification models.
        """
        if image is None:
            return None, "Please upload an image."
        
        if not project_name or not model_name:
            return image, "Please select a Project and Model."
            
        model_path = self.train_root_dir / project_name / model_name / "weights" / "best.pt"
        if not model_path.exists():
            return image, f"Model not found at {model_path}"
            
        try:
            # Detect model type by checking if model_name contains "-cls"
            is_classification = "-cls" in model_name.lower()
            
            # Load model
            model = YOLO(model_path)
            
            # Run inference based on model type
            if is_classification:
                # Classification: No conf/iou parameters
                print(f"Running classification prediction on model: {model_name}")
                results = model.predict(image, device=0, verbose=False)
                
                # For classification, get top-1 prediction
                if results and len(results) > 0:
                    probs = results[0].probs
                    top1_idx = probs.top1
                    top1_conf = float(probs.top1conf)
                    label = model.names[top1_idx]
                    
                    # Create a simple visualization with the label
                    output_image = image.copy()
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(output_image)
                    
                    # Draw label on image
                    text = f"{label}: {top1_conf:.3f}"
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
                    except:
                        font = ImageFont.load_default()
                    
                    # Draw text with background
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Position at top-left
                    x, y = 10, 10
                    draw.rectangle([x, y, x + text_width + 10, y + text_height + 10], fill=(0, 255, 0))
                    draw.text((x + 5, y + 5), text, fill=(0, 0, 0), font=font)
                    
                    # Format detailed output
                    json_results = {
                        "model_type": "classification",
                        "predicted_class": label,
                        "confidence": top1_conf,
                        "top5_predictions": [
                            {"class": model.names[i], "confidence": float(probs.data[i])}
                            for i in probs.top5
                        ]
                    }
                    
                    return output_image, json.dumps(json_results, indent=2)
                else:
                    return image, "No classification results"
                    
            else:
                # Detection/Segmentation: Use conf and iou parameters
                print(f"Running detection/segmentation prediction on model: {model_name}")
                results = model.predict(image, conf=conf_threshold, iou=iou_threshold, device=0, verbose=False)
                
                # Plot results on the image (returns numpy array in BGR)
                res_plotted = results[0].plot() 
                output_image = Image.fromarray(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
                
                # Format detailed output
                details = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    score = float(box.conf[0])
                    details.append(f"{label}: {score:.3f}")
                    
                json_results = {
                    "model_type": "detection/segmentation",
                    "detections_count": len(results[0].boxes),
                    "detections": details
                }
                
                return output_image, json.dumps(json_results, indent=2)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return image, f"Error: {e}"
    
    def create_dataset_html(self):
        """Create HTML representation of dataset buttons with scrollable container"""
        datasets = self.get_all_datasets()
        if not datasets:
            return "<p style='text-align: center; color: #9a3412; font-style: italic;'>No datasets found in settings.json.</p>"
        
        # Create scrollable container that shows max 5 items
        html = """
        <div style='max-height: 400px; overflow-y: auto; border: 2px solid #fb923c; 
                    border-radius: 12px; padding: 10px; background: linear-gradient(135deg, #000000 0%, #000000 100%);'>
            <div style='display: flex; flex-direction: column; gap: 12px;'>
        """
        
        for i, dataset in enumerate(datasets):
            html += f"""
            <div class='dataset-card' style='display: flex; justify-content: space-between; align-items: center; 
                        padding: 15px; border: 2px solid #fb923c; border-radius: 12px; 
                        background: linear-gradient(135deg, #000000 0%, #000000 100%);;
                        box-shadow: 0 2px 8px rgba(234, 88, 12, 0.1);
                        transition: all 0.3s ease;'
                        onmouseover='this.style.transform="translateY(-2px)"; this.style.boxShadow="0 4px 12px rgba(234, 88, 12, 0.2)";'
                        onmouseout='this.style.transform="translateY(0)"; this.style.boxShadow="0 2px 8px rgba(234, 88, 12, 0.1)";'>
                <div style='flex-grow: 1;'>
                    <div style='display: flex; align-items: center; margin-bottom: 8px;'>
                        <span style='background: linear-gradient(135deg, #ea580c 0%, #dc2626 100%);
                                    color: white; padding: 4px 8px; border-radius: 16px; 
                                    font-size: 12px; font-weight: bold; margin-right: 10px;'>
                            #{i+1}
                        </span>
                        <strong style='color: #9a3412; font-size: 16px;'>{dataset['name']}</strong>
                    </div>
                    <div style='color: #c2410c; font-size: 12px; line-height: 1.4;'>
                        <div style='margin-bottom: 2px;'>
                            📁 <strong>Path:</strong> {dataset.get('path', 'N/A')}
                        </div>
                    </div>
                </div>
                <div style='margin-left: 15px;'>
                    <div style='width: 8px; height: 40px; background: linear-gradient(135deg, #fb923c 0%, #ea580c 100%); 
                            border-radius: 4px; opacity: 0.6;'></div>
                </div>
            </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        # Add info about scrolling if there are more than 5 datasets
        if len(datasets) > 5:
            html += f"""
            <div style='text-align: center; margin-top: 10px; color: #c2410c; font-size: 12px; font-style: italic;'>
                📜 Showing {len(datasets)} datasets - scroll to view all
            </div>
            """
        
        return html

    def upload_dataset_by_zip(self, zip_file_path):

        if zip_file_path is None:
            return False, "No file was uploaded. Please upload a ZIP file."
        
        # Check if the uploaded file is a ZIP file
        if not zipfile.is_zipfile(zip_file_path):
            return False, "The uploaded file is not a valid ZIP file. Please upload a .zip archive."
        
        try:
            project_name = zip_file_path.split('/')[-1].replace('.zip', '')
            output_dir = os.path.join(self.datasets_dir, project_name)
            
            # Use smart extraction that auto-flattens nested structures
            self.extract_and_flatten_zip(zip_file_path, output_dir)
            

            # Update the config with the new dataset
            if "datasets" not in self.config:
                self.config["datasets"] = []
            self.config["datasets"].append({"name": project_name, "path": output_dir})
            self.save_config()

            return True, f"Successfully unzipped the file to '{output_dir}'.\n\nExtracted files:\n" + "\n".join([f"- {file}" for file in os.listdir(output_dir)])
        except Exception as e:
            return False, f"An error occurred while unzipping the file: {e}"

    def get_pretrained_models(self, format_name=None):
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
        
        models_dir = Path(f"models/pre_trained/{sub_dir}")
        if not models_dir.exists():
            return []
        
        return [f.name for f in models_dir.glob("*.pt")]
    
    def load_own_models(self):
        """
        Load custom models from settings.json.
        Returns: List of model dictionaries with name and path
        """
        own_models = self.config.get("own_models", [])
        if not isinstance(own_models, list):
            self.config["own_models"] = []
            self.save_config()
            return []
        return own_models
    
    def save_own_model(self, model_name, model_file_path):
        """
        Save uploaded model to .gradio/own_models/ and update settings.json.
        
        Args:
            model_name: str - Name for the model
            model_file_path: str - Path to uploaded .pt file
            
        Returns:
            tuple: (success: bool, message: str)
        """
        if not model_name or not model_name.strip():
            return False, "❌ Please provide a model name"
        
        if not model_file_path or not os.path.exists(model_file_path):
            return False, "❌ Please upload a valid .pt file"
        
        model_name = model_name.strip()
        
        # Create own_models directory
        own_models_dir = Path(self.datasets_dir) / "own_models"
        own_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Destination path
        dest_filename = f"{model_name}.pt"
        dest_path = own_models_dir / dest_filename
        
        try:
            # Copy file to destination
            shutil.copy2(model_file_path, dest_path)
            
            # Initialize own_models if not exists
            if "own_models" not in self.config:
                self.config["own_models"] = []
            
            # Check if model already exists
            existing_models = self.config["own_models"]
            existing_idx = next((i for i, m in enumerate(existing_models) if m.get("name") == model_name), -1)
            
            model_entry = {
                "name": model_name,
                "path": str(dest_path)
            }
            
            if existing_idx >= 0:
                # Update existing
                self.config["own_models"][existing_idx] = model_entry
                message = f"✅ Model '{model_name}' updated successfully"
            else:
                # Add new
                self.config["own_models"].append(model_entry)
                message = f"✅ Model '{model_name}' uploaded successfully"
            
            self.save_config()
            return True, message
            
        except Exception as e:
            return False, f"❌ Error saving model: {e}"
    
    def delete_own_model(self, model_name):
        """
        Delete model file and remove from settings.json.
        
        Args:
            model_name: str - Name of model to delete
            
        Returns:
            tuple: (success: bool, message: str)
        """
        if not model_name:
            return False, "❌ Please select a model to delete"
        
        own_models = self.config.get("own_models", [])
        model_entry = next((m for m in own_models if m.get("name") == model_name), None)
        
        if not model_entry:
            return False, f"❌ Model '{model_name}' not found"
        
        # Delete physical file
        model_path = Path(model_entry.get("path", ""))
        if model_path.exists():
            try:
                os.remove(model_path)
            except Exception as e:
                return False, f"❌ Error deleting file: {e}"
        
        # Remove from config
        self.config["own_models"] = [m for m in own_models if m.get("name") != model_name]
        self.save_config()
        
        return True, f"✅ Model '{model_name}' deleted successfully"
    
    def get_own_model_details(self, model_name):
        """
        Load model and return metadata.
        
        Args:
            model_name: str - Name of model
            
        Returns:
            dict: Model details or error message
        """
        if not model_name:
            return {"Status": "No model selected"}
        
        own_models = self.config.get("own_models", [])
        model_entry = next((m for m in own_models if m.get("name") == model_name), None)
        
        if not model_entry:
            return {"Error": "Model not found"}
        
        model_path = model_entry.get("path", "")
        
        if not os.path.exists(model_path):
            return {"Error": "Model file not found"}
        
        try:
            # Load YOLO model
            model = YOLO(model_path)
            
            # Extract metadata
            params = sum(p.numel() for p in model.model.parameters())
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            # Get class names
            classes = {}
            if hasattr(model, 'names') and model.names:
                classes = {str(k): v for k, v in model.names.items()}
            
            return {
                "Model": model_name,
                "Params": f"{params:,}",
                "Size": f"{size_mb:.2f} MB",
                "Classes": classes if classes else "N/A"
            }
            
        except Exception as e:
            return {"Error": f"Failed to load model: {e}"}
    
    def predict_with_own_model(self, model_name, image, conf_threshold=0.25, iou_threshold=0.45):
        """
        Run prediction using custom model.
        
        Args:
            model_name: str - Name of custom model
            image: PIL Image
            conf_threshold: float - Confidence threshold
            iou_threshold: float - IOU threshold
            
        Returns:
            tuple: (output_image, detection_details_json)
        """
        if image is None:
            return None, json.dumps({"error": "Please upload an image"}, indent=2)
        
        if not model_name:
            return image, json.dumps({"error": "Please select a model"}, indent=2)
        
        own_models = self.config.get("own_models", [])
        model_entry = next((m for m in own_models if m.get("name") == model_name), None)
        
        if not model_entry:
            return image, json.dumps({"error": "Model not found"}, indent=2)
        
        model_path = model_entry.get("path", "")
        
        if not os.path.exists(model_path):
            return image, json.dumps({"error": "Model file not found"}, indent=2)
        
        try:
            # Load model
            model = YOLO(model_path)
            
            # Detect model type (detection, segmentation, or classification)
            is_classification = "-cls" in model_name.lower() or (hasattr(model, 'task') and model.task == 'classify')
            
            if is_classification:
                # Classification prediction
                results = model.predict(image, device=0, verbose=False)
                
                if results and len(results) > 0:
                    probs = results[0].probs
                    top1_idx = probs.top1
                    top1_conf = float(probs.top1conf)
                    label = model.names[top1_idx]
                    
                    # Create visualization
                    output_image = image.copy()
                    draw = ImageDraw.Draw(output_image)
                    
                    text = f"{label}: {top1_conf:.3f}"
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
                    except:
                        font = ImageFont.load_default()
                    
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    x, y = 10, 10
                    draw.rectangle([x, y, x + text_width + 10, y + text_height + 10], fill=(0, 255, 0))
                    draw.text((x + 5, y + 5), text, fill=(0, 0, 0), font=font)
                    
                    json_results = {
                        "model_type": "classification",
                        "predicted_class": label,
                        "confidence": top1_conf,
                        "top5_predictions": [
                            {"class": model.names[i], "confidence": float(probs.data[i])}
                            for i in probs.top5
                        ]
                    }
                    
                    return output_image, json.dumps(json_results, indent=2)
                else:
                    return image, json.dumps({"status": "No classification results"}, indent=2)
            
            else:
                # Detection/Segmentation prediction
                results = model.predict(image, conf=conf_threshold, iou=iou_threshold, device=0, verbose=False)
                
                # Plot results
                res_plotted = results[0].plot()
                output_image = Image.fromarray(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
                
                # Format details
                details = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    score = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    details.append({
                        "label": label,
                        "confidence": score,
                        "bbox": xyxy
                    })
                
                json_results = {
                    "model_type": "detection/segmentation",
                    "detections_count": len(results[0].boxes),
                    "detections": details
                }
                
                return output_image, json.dumps(json_results, indent=2)
        
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return image, json.dumps({"error": str(e)}, indent=2)

    def get_random_sample_images(self, task_id, count=4):
        """
        Extracts up to 'count' random images from the CVAT task zip.
        Downloads the zip if it doesn't exist.
        Returns: List of PIL Image objects
        """
        if not task_id:
            return None
        
        # 1. Check/Download Zip
        raw_zip_path = Path(self.datasets_dir) / f"cvat_task_{task_id}.zip"
        
        if not raw_zip_path.exists():
            # Attempt download (simplified version of download logic)
            try:
                url = self.config.get("cvat_url")
                username = self.config.get("cvat_username")
                password = self.config.get("cvat_password")
                host = url.split('/projects')[0].split('/tasks')[0].split('jobs')[0].rstrip('/')
                
                print(f"Downloading Task {task_id} for preview...")
                with make_client(host, credentials=(username, password)) as client:
                    task = client.tasks.retrieve(int(task_id))
                    raw_zip_path.parent.mkdir(parents=True, exist_ok=True)
                    task.export_dataset(
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
                with zip_ref.open(random_files[0]) as file: # Keep zip open? No, extracting logic one by one
                    pass

                # We need to open them all.
                for fname in random_files:
                     with zip_ref.open(fname) as file:
                        img_data = file.read()
                        images.append(Image.open(io.BytesIO(img_data)))
                
                return images

        except Exception as e:
            print(f"Error extracting preview images: {e}")
            return []

    def preview_augmentation(self, images, **kwargs):
        """
        Applies OpenCV-based augmentations to a list of PIL images for preview.
        Handles Mosaic (4 images), Mixup (2 images), etc.
        Returns: PIL Image (Result)
        """
        if not images or images[0] is None:
            return None
            
        # Primary image for single-img transforms
        # For Mosaic, we construct a new primary from 4 images
        
        # Convert all to CV2 BGR
        cv_images = []
        for img in images:
            if img is not None:
                cv_images.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        
        if not cv_images: return None

        img_cv = cv_images[0] # Default primary
        h, w = img_cv.shape[:2]

        try:
            # --- Multi-Image Augmentations (Mosaic, Mixup) ---
            # Priority: Mosaic > Mixup/Cutmix > Single Image
            
            mosaic_prob = kwargs.get('mosaic', 0.0)
            mixup_prob = kwargs.get('mixup', 0.0)
            cutmix_prob = kwargs.get('cutmix', 0.0)
            copy_paste_prob = kwargs.get('copy_paste', 0.0)

            # 1. Mosaic (Requires 4 images)
            if mosaic_prob > 0.5 and len(cv_images) >= 4:
                # Create 2x2 grid
                # Target size: 2 * w, 2 * h (or keep original size? YOLO mosaic creates large canvas then crops)
                # For preview, let's make a canvas 2x size then center crop or resize back.
                # Simplified: Resize all to w/2, h/2 and place
                
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                xc, yc = w // 2, h // 2 # Center point (simulated)

                # Top-Left
                img_tl = cv2.resize(cv_images[0], (xc, yc))
                canvas[0:yc, 0:xc] = img_tl
                
                # Top-Right
                img_tr = cv2.resize(cv_images[1], (w-xc, yc))
                canvas[0:yc, xc:w] = img_tr
                
                # Bottom-Left
                img_bl = cv2.resize(cv_images[2], (xc, h-yc))
                canvas[yc:h, 0:xc] = img_bl
                
                # Bottom-Right
                img_br = cv2.resize(cv_images[3], (w-xc, h-yc))
                canvas[yc:h, xc:w] = img_br
                
                img_cv = canvas
            
            # 2. Mixup (Requires 2 images) - Blending
            elif mixup_prob > 0.5 and len(cv_images) >= 2:
                im1 = img_cv
                im2 = cv2.resize(cv_images[1], (w, h))
                # Alpha blend
                img_cv = cv2.addWeighted(im1, 0.5, im2, 0.5, 0)

            # 3. CutMix (Requires 2 images) - Patch Replacement
            elif cutmix_prob > 0.5 and len(cv_images) >= 2:
                im1 = img_cv
                im2 = cv2.resize(cv_images[1], (w, h))
                
                # Cut random patch from im2 and paste to im1
                # Size: 50% width/height
                pw, ph = w // 2, h // 2
                x = random.randint(0, w - pw)
                y = random.randint(0, h - ph)
                
                img_cv[y:y+ph, x:x+pw] = im2[y:y+ph, x:x+pw]

            # 4. Copy-Paste (Requires Segm normally, simplified here as Patch Paste)
            elif copy_paste_prob > 0.5 and len(cv_images) >= 2:
                 # Paste a smaller object-like patch (e.g. 20%) from im2 to random loc on im1
                 im2 = cv2.resize(cv_images[1], (w, h))
                 pw, ph = int(w * 0.2), int(h * 0.2)
                 
                 # Source crop
                 sx = random.randint(0, w - pw)
                 sy = random.randint(0, h - ph)
                 patch = im2[sy:sy+ph, sx:sx+pw]
                 
                 # Dest loc
                 dx = random.randint(0, w - pw)
                 dy = random.randint(0, h - ph)
                 
                 # Simple paste (no alpha for bbox simulation)
                 img_cv[dy:dy+ph, dx:dx+pw] = patch


            # --- Single Image Augmentations (Applied to result of above) ---
            # 1. HSV Augmentation
            # hsv_h, hsv_s, hsv_v are fractions (0.0 - 1.0)
            if any(k in kwargs for k in ['hsv_h', 'hsv_s', 'hsv_v']):
                h_gain = kwargs.get('hsv_h', 0.015)
                s_gain = kwargs.get('hsv_s', 0.7)
                v_gain = kwargs.get('hsv_v', 0.4)
                
                # Only apply if significantly non-zero (YOLO defaults are small)
                # For preview, we simulate a random variation within the gain range
                # Or just apply the gain to show "max effect"? 
                # User wants to see effect. Let's apply a deterministic shift proportional to gain 
                # to show "what could happen".
                
                img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV).astype(np.float32) # Hue is 0-179
                
                # Simulate a +50% of the range shift for visualization
                hue_shift = (h_gain * 179) * 0.5 
                sat_shift = (s_gain * 255) * 0.5
                val_shift = (v_gain * 255) * 0.5
                
                img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_shift) % 180
                img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] + sat_shift, 0, 255)
                img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] + val_shift, 0, 255)
                
                img_cv = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # 2. Geometric - Flip LR
            if kwargs.get('fliplr', 0.0) > 0.5: # Show flip if prob > 50%
                img_cv = cv2.flip(img_cv, 1)

            # 3. Geometric - Flip UD
            if kwargs.get('flipud', 0.0) > 0.5:
                img_cv = cv2.flip(img_cv, 0)
            
            # --- Geometric Transforms (Consolidated) ---
            # We build an affine/perspective matrix to apply rotation, scale, shear, translation, perspective together.
            
            deg = kwargs.get('degrees', 0.0)
            trans = kwargs.get('translate', 0.1)
            scale_gain = kwargs.get('scale', 0.5)
            shear_deg = kwargs.get('shear', 0.0)
            persp = kwargs.get('perspective', 0.0)

            # Only proceed if any geometric param is non-trivial
            if deg != 0 or trans != 0 or scale_gain != 0 or shear_deg != 0 or persp != 0:
                
                # Center of image
                C = np.eye(3)
                C[0, 2] = -w / 2
                C[1, 2] = -h / 2

                # Rotation and Scale
                # YOLO scale is +/- gain. Let's simulate a random scale in [1-scale, 1+scale]
                # For preview visualization, we use a fixed "demo" value like 1.0 + (scale * 0.5)
                # or just the mean effect. Let's show a distinct effect.
                s = 1.0 + (scale_gain * 0.5) # Zoom in a bit
                
                R = np.eye(3)
                a = np.deg2rad(deg)
                R[0, 0] = R[1, 1] = s * np.cos(a)
                R[0, 1] = -s * np.sin(a)
                R[1, 0] = s * np.sin(a)
                
                # Shear
                S = np.eye(3)
                S[0, 1] = np.tan(np.deg2rad(shear_deg)) # x-shear

                # Translation
                T = np.eye(3)
                T[0, 2] = w / 2 + (trans * w * 0.5) # Shift by half the allowed translation range
                T[1, 2] = h / 2 + (trans * h * 0.5)

                # Combined Affine Matrix
                M = T @ S @ R @ C  # Order: Center -> Rotate/Scale -> Shear -> Translate back + offset
                
                # Perspective
                if persp != 0:
                    P = np.eye(3)
                    P[2, 0] = persp * 0.001 # Small p effect
                    P[2, 1] = persp * 0.001
                    
                    M = P @ M # Apply perspective after affine
                    
                    img_cv = cv2.warpPerspective(img_cv, M, (w, h), borderValue=(114, 114, 114))
                else:
                    img_cv = cv2.warpAffine(img_cv, M[:2], (w, h), borderValue=(114, 114, 114))

            # 6. Mosaic (Simulated)
            # handled above in Multi-Image section
            pass
            
            # 7. Erasing
            erase_prob = kwargs.get('erasing', 0.0)
            if erase_prob > 0:
                # Draw a random black box
                ex = int(w * 0.2)
                ey = int(h * 0.2)
                ew = int(w * 0.2)
                eh = int(h * 0.2)
                cv2.rectangle(img_cv, (ex, ey), (ex+ew, ey+eh), (128, 128, 128), -1)

            # Convert back to RGB for PIL
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_rgb)
            
        except Exception as e:
            print(f"Augmentation preview error: {e}")
            return images[0] if images else None
        
    def start_training(self, formatted_path, model_name, epochs, imgsz=640, manual_aug=False, format_name="Ultralytics YOLO Detection 1.0", **kwargs):
        """
        Orchestrates the training process:
        1. Locate and extract the formatted dataset zip
        2. Call train.py
        """
        if not formatted_path:
            return "❌ Error: No dataset selected."
        if not model_name:
            return "❌ Error: No model selected."
            
        print(f"Starting training: Dataset Path={formatted_path}, Model={model_name}, Epochs={epochs}, ImgSz={imgsz}, ManualAug={manual_aug}")
        
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
        
        # 1. Get dataset path
        formatted_dfs = self.config.get("formatted_datasets", [])
        dataset_entry = next((d for d in formatted_dfs if d["path"] == formatted_path), None)
        
        # Handle both registered datasets and uploaded datasets
        if dataset_entry:
            # Dataset from CVAT task (registered in config)
            formatted_dataset_name = dataset_entry["name"]
        else:
            # Uploaded dataset (not in config) - derive name from filename
            formatted_dataset_name = Path(formatted_path).stem.replace("formatted_", "")
            print(f"Using uploaded dataset: {formatted_dataset_name}")
            
        zip_path = Path(formatted_path)
        if not zip_path.exists():
            return f"❌ Error: Zip file not found at {zip_path}"
            
        # 2. Extract to Dataset folder (using utils.ProjectManager conventions)
        # We'll use the dataset name as the project name
        project_name = formatted_dataset_name
        # ProjectManager expects data in specific location.
        # We need to extract to /home/intern/Gitlab/pipeline/Dataset/<project_name>
        # Let's manually handle extraction to ensure it matches what train.py expects via ProjectManager
        
        target_dir = Path("Dataset") / project_name
        
        try:
            # Clean existing if needed or just overwrite? ZipFile extractall overwrites.
            if target_dir.exists():
                shutil.rmtree(target_dir)
            
            # Use smart extraction that auto-flattens nested structures
            self.extract_and_flatten_zip(zip_path, target_dir)
                
            print(f"Extracted dataset to {target_dir}")
            
            # --- Rewrite paths to absolute paths ---
            abs_target_dir = target_dir.resolve()
            
            # 1. Update data.yaml
            yaml_path = target_dir / "data.yaml"
            if yaml_path.exists():
                with open(yaml_path, 'r') as f:
                    yaml_lines = f.readlines()
                
                new_yaml_lines = []
                for line in yaml_lines:
                    if line.strip().startswith("train:"):
                        new_yaml_lines.append(f"train: {abs_target_dir / 'Train.txt'}\n")
                    elif line.strip().startswith("val:"):
                        new_yaml_lines.append(f"val: {abs_target_dir / 'Validation.txt'}\n")
                    elif line.strip().startswith("test:"):
                        new_yaml_lines.append(f"test: {abs_target_dir / 'Test.txt'}\n")
                    else:
                        new_yaml_lines.append(line)
                
                with open(yaml_path, 'w') as f:
                    f.writelines(new_yaml_lines)
            
            # 2. Update Train.txt and Validation.txt
            for txt_name in ["Train.txt", "Validation.txt", "Test.txt"]:
                txt_path = target_dir / txt_name
                if txt_path.exists():
                    with open(txt_path, 'r') as f:
                        lines = f.readlines()
                    
                    new_lines = []
                    for line in lines:
                        line = line.strip()
                        if line.startswith("./"):
                            # Replace ./ with absolute path
                            new_lines.append(str(abs_target_dir / line[2:]) + "\n")
                        else:
                            # Fallback if it doesn't start with ./ (e.g. already absolute or relative without dot)
                            # Assuming our formatter writes ./
                            new_lines.append(line + "\n")
                    
                    with open(txt_path, 'w') as f:
                        f.writelines(new_lines)
            
        except Exception as e:
            return f"❌ Error extracting dataset: {e}"
            
        # 3. Call train.py
        try:
            import train
            
            # Construct absolute model path
            sub_dir = "detection"
            if format_name and ("Segmentation" in format_name):
                 sub_dir = "segmentation"
            elif format_name and ("Classification" in format_name):
                 sub_dir = "classification"

            model_path = str(Path(f"models/pre_trained/{sub_dir}") / model_name)
            
            success, msg = train.run_training(
                project_name=project_name,
                model_path=model_path,
                epochs=int(epochs),
                imgsz=int(imgsz),
                manual_aug=manual_aug,
                aug_params=aug_params,
                format_name=format_name
            )
            
            if success:
                try:
                    # Check if file is in the root .gradio dir (not own_formatted)
                    # and starts with formatted_task_ or manually matches
                    root_dir = Path(self.datasets_dir).resolve()
                    parent_dir = zip_path.parent.resolve()
                    
                    print(f"DEBUG Cleanup: ZIP={zip_path}, Parent={parent_dir}, Root={root_dir}")
                    print(f"DEBUG Cleanup: Exists={zip_path.exists()}, IsInRoot={parent_dir == root_dir}")

                    is_in_root = parent_dir == root_dir
                    if is_in_root and zip_path.exists():
                        print(f"🧹 Cleaning up temporary zip: {zip_path}")
                        zip_path.unlink()

                        # Remove from config
                        formatted_datasets = self.config.get("formatted_datasets", [])
                        formatted_datasets = [d for d in formatted_datasets if d["path"] != formatted_path]
                        self.config["formatted_datasets"] = formatted_datasets
                        self.save_config()


                except Exception as cleanup_err:
                    print(f"⚠️ Cleanup warning: {cleanup_err}")

                return f"✅ {msg}"
            else:
                return f"❌ {msg}"
                
        except Exception as e:
            return f"❌ Error invoking training: {e}"
        
    def get_all_datasets(self, name=None):
        # A list of datasets from the configuration, each element is a dict with 'name' and 'path'
        return self.config.get("datasets", [])

    def get_all_formatted_datasets(self, name=None):
        # A list of datasets from the configuration, each element is a dict with 'name' and 'path'
        return self.config.get("formatted_datasets", [])

    def get_all_own_datasets(self, name=None):
        # A list of user-uploaded formatted datasets from the configuration
        return self.config.get("own_datasets", [])

    def select_formatted_dataset_path(self, name):
        """
        Retrieves the path of a formatted dataset by its name and returns statistics.
        """
        datasets = self.config.get("formatted_datasets", [])
        dataset = next((d for d in datasets if d["name"] == name), None)
        
        if not dataset:
             return (
                f"<div style='padding: var(--size-2); border: 1px solid var(--block-border-color); "
                f"background: var(--input-background-fill); border-radius: var(--container-radius); "
                f"color: var(--body-text-color); min-height: 80px;'>"
                f"Formatted Dataset '<b>{name}</b>' not found."
                f"</div>"
            )

        zip_path_str = dataset.get('path', '')
        zip_path = Path(zip_path_str)
        
        stats = {
            "train_images": 0, "val_images": 0,
            "train_labels": 0, "val_labels": 0,
            "classes": "N/A"
        }

        if zip_path.exists():
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    file_list = z.namelist()
                    
                    # Count files based on paths
                    for f in file_list:
                        if f.endswith('/'): continue # Skip directories
                        
                        lower_f = f.lower()
                        # Adjust detection logic based on standard structure
                        if "images/train/" in lower_f: stats["train_images"] += 1
                        elif "images/validation/" in lower_f: stats["val_images"] += 1
                        elif "labels/train/" in lower_f: stats["train_labels"] += 1
                        elif "labels/validation/" in lower_f: stats["val_labels"] += 1
                        
                    # Parse data.yaml for classes
                    try:
                        # Find data.yaml (could be at root or nested?) Assuming root based on formatting logic
                        # But formatting logic might put it in a subfolder if zipped incorrectly, 
                        # however our formatting logic puts it at root of archive usually 
                        # (shutil.make_archive of temp_build_dir).
                        # Let's check for 'data.yaml' or any '*/data.yaml'
                        yaml_file = next((f for f in file_list if f.endswith('data.yaml')), None)
                        
                        if yaml_file:
                            with z.open(yaml_file) as yf:
                                content = yf.read().decode('utf-8')
                                # Simple parsing to avoid PyYAML dependency inside this function if not imported
                                # But we can just search for 'nc:'
                                lines = content.split('\n')
                                for line in lines:
                                    if line.strip().startswith('nc:'):
                                        stats["classes"] = line.split(':')[1].strip()
                                        break
                                else:
                                    # Fallback: check names list length
                                    # This is complex to parse via string split, keeping it simple for now
                                    pass
                    except Exception as e:
                        print(f"Error reading yaml: {e}")
                        stats["classes"] = "Error"
                        
            except Exception as e:
                print(f"Error reading zip: {e}")
                return (
                    f"<div style='padding: var(--size-2); border: 1px solid var(--block-border-color); "
                    f"background: var(--input-background-fill); border-radius: var(--container-radius); "
                    f"color: var(--body-text-color); min-height: 80px;'>"
                    f"Error reading dataset file: {e}"
                    f"</div>"
                )
        
        return (
            f"<div style='padding: var(--size-2); border: 1px solid var(--block-border-color); "
            f"background: var(--input-background-fill); border-radius: var(--container-radius); "
            f"color: var(--body-text-color); min-height: 80px; font-family: monospace;'>"
            f"<b>Selected:</b> {name}<br>"
            f"<b>Path:</b> {zip_path_str}<br>"
            f"<hr style='margin: 5px 0; border-color: var(--border-color-primary);'>"
            f"<b>Images:</b> Train: {stats['train_images']} | Val: {stats['val_images']}<br>"
            f"<b>Labels:</b> Train: {stats['train_labels']} | Val: {stats['val_labels']}<br>"
            f"<b>Classes:</b> {stats['classes']}"
            f"</div>"
        )

    def get_dataset_by_name(self, name):
        """
        Retrieves a single dataset's details by its name.
        """
        dataset = next((d for d in self.config.get("datasets", []) if d["name"] == name), None)
        if dataset: return True, dataset
        return False, f"Dataset '{name}' not found."    
    
    def remove_dataset(self, name):
        """
        Removes a dataset from the configuration and also deletes its directory from the file system.
        """
        datasets = self.config.get("datasets", [])
        success, dataset_to_remove = self.get_dataset_by_name(name)
        
        if not success:
            return False, f"Dataset '{name}' not found."

        # Attempt to remove the directory
        dataset_path = dataset_to_remove.get("path")
        if dataset_path and os.path.exists(dataset_path):
            message = f"Attempting to delete directory: {dataset_path}"
            try:
                shutil.rmtree(dataset_path)
                message = f"Successfully deleted directory: {dataset_path}"
            except OSError as e:
                message = f"Error deleting directory {dataset_path}: {e}"
                return False, message
        else:
            message = f"Directory for '{name}' not found on disk, only removing from configuration."
            
        # Remove the dataset from the configuration list
        self.config["datasets"] = [d for d in datasets if d.get("name") != name]
        self.save_config()

        return True, f"Dataset '{name}' removed from configuration successfully."

    def save_config(self):
        """
        Saves the current configuration dictionary to the settings.json file.
        """
        try:
            # Open the file in write mode ('w') and write the dictionary as JSON
            with open(self.json_path, 'w') as config_file:
                json.dump(self.config, config_file, indent=4)

            
        except Exception as e:
            print(f"An error occurred while saving the configuration: {e}")

    def scan_for_videos(self, dataset_name):
        """
        Scans the dataset for video files and returns a list of dictionaries.
        Each dictionary contains: 'filename', 'duration', 'duration_sec'.
        """
        success, dataset = self.get_dataset_by_name(dataset_name)
        if not success:
            return []
            
        dataset_path = dataset.get("path", "")
        if not os.path.exists(dataset_path):
            return []
            
        video_files = []
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        
        for file in os.listdir(dataset_path):
            if file.lower().endswith(video_extensions):
                full_path = os.path.join(dataset_path, file)
                try:
                    cap = cv2.VideoCapture(full_path)
                    if not cap.isOpened():
                        continue
                        
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration_sec = frame_count / fps if fps > 0 else 0
                    
                    # specific formatting for duration
                    minutes = int(duration_sec // 60)
                    seconds = int(duration_sec % 60)
                    duration_str = f"{minutes}m {seconds}s"
                    
                    video_files.append({
                        "Video Name": file,
                        "Duration": duration_str,
                        "Extraction Interval": "1s" # Default value
                    })
                    cap.release()
                except Exception as e:
                    print(f"Error reading video {file}: {e}")
                    
        return video_files

    def parse_interval(self, interval_str):
        """
        Parses interval string like '1s', '10s', '1m', '0.5m', '100ms'.
        Returns seconds as float.
        """
        interval_str = interval_str.strip().lower()
        if interval_str.endswith('ms'):
            return float(interval_str[:-2]) / 1000.0
        elif interval_str.endswith('s'):
            return float(interval_str[:-1])
        elif interval_str.endswith('m'):
            return float(interval_str[:-1]) * 60.0
        else:
            # Assume seconds if no unit
            try:
                return float(interval_str)
            except ValueError:
                return 1.0 # Default fallback

    def extract_frames_from_dataset(self, dataset_name, video_config_df, interval_val=1.0):
        """
        Creates a temporary dataset with extracted frames + original images.
        video_config_df is a pandas DataFrame or list containing video info.
        interval_val: float (seconds)
        """
        success, dataset = self.get_dataset_by_name(dataset_name)
        if not success:
            return False, "Dataset not found"

        source_path = dataset.get("path")
        temp_name = f"{dataset_name}_temp_frames"
        temp_dir = os.path.join(self.datasets_dir, temp_name)
        
        # 1. Create temp directory (clean if exists)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        # 2. Copy existing images
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        for file in os.listdir(source_path):
            if file.lower().endswith(image_extensions):
                shutil.copy2(os.path.join(source_path, file), os.path.join(temp_dir, file))
                
        # 3. Process Videos
        # Convert df to list of dicts if needed (pandas df to records)
        # We need to know column types. Assumed order: Name, Duration, (Interval ignored)
        
        # If input is pandas DataFrame
        import pandas as pd
        if isinstance(video_config_df, pd.DataFrame):
            config_records = video_config_df.to_dict('records')
        else:
            if isinstance(video_config_df, list):
                if len(video_config_df) > 0 and isinstance(video_config_df[0], list):
                     config_records = [
                         {"Video Name": r[0], "Duration": r[1]} 
                         for r in video_config_df
                     ]
                else:
                    config_records = video_config_df
            else:
                 config_records = []

        # Calculate target interval in seconds
        try:
            val = float(interval_val)
        except:
            val = 1.0
            
        target_interval_sec = val

        for record in config_records:
            video_name = record.get("Video Name")
            interval_sec = target_interval_sec
            
            video_path = os.path.join(source_path, video_name)
            if not os.path.exists(video_path):
                continue
                
            try:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0: continue
                
                step_frames = int(fps * interval_sec)
                if step_frames < 1: step_frames = 1
                
                frame_idx = 0
                count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    if frame_idx % step_frames == 0:
                        # Save frame
                        frame_name = f"{os.path.splitext(video_name)[0]}_frame_{count:04d}.jpg"
                        cv2.imwrite(os.path.join(temp_dir, frame_name), frame)
                        count += 1
                        
                    frame_idx += 1
                cap.release()
            except Exception as e:
                print(f"Failed to extract {video_name}: {e}")

        # 4. Register new dataset in config
        # Check if already exists in config, update it
        existing_idx = next((i for i, d in enumerate(self.config["datasets"]) if d["name"] == temp_name), -1)
        new_entry = {"name": temp_name, "path": temp_dir, "is_temp": True}
        
        if existing_idx >= 0:
            self.config["datasets"][existing_idx] = new_entry
        else:
            self.config["datasets"].append(new_entry)
            
        self.save_config()
        self.select_dataset(temp_name) # Switch to this dataset
        
        return True, temp_name

    def cleanup_temp_datasets(self):
        """
        Removes any datasets with names ending in '_temp_frames'.
        Used to clean up temporary datasets created for video inference.
        """
        temp_suffix = "_temp_frames"
        datasets_to_remove = [d["name"] for d in self.config.get("datasets", []) if d["name"].endswith(temp_suffix)]
        
        removed_count = 0
        for name in datasets_to_remove:
            success, msg = self.remove_dataset(name)
            if success:
                removed_count += 1
                # print(f"Cleaned up temp dataset: {name}") # Optional logging
            else:
                print(f"Failed to clean up temp dataset {name}: {msg}")
                
        return removed_count

    def draw_bounding_boxes(self, image, detections, threshold=0.3):
        """
        Draw bounding boxes on the image with labels and confidence scores.
        
        Args:
            image: PIL Image
            detections: Detection results from the model
            labels: List of labels for each detection
            threshold: Confidence threshold for displaying boxes
        
        Returns:
            PIL Image with bounding boxes drawn
        """
        # Create a copy of the image to draw on
        image_with_boxes = image.copy()
        draw = ImageDraw.Draw(image_with_boxes)
        
        # Try to load a font (fallback to default if not available)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Colors for different objects
        colors = [
            "red"#, "blue", "green", "yellow", "purple", "orange", "pink", "brown",
            #"gray", "cyan", "magenta", "lime", "indigo", "violet", "turquoise"
        ]
        colors = ['red']
        
        width, height = image.size
        
        # Process each detection
        for i, det in enumerate(detections):
            label = det['label']
            score = float(det['score'])
            box = det['box']  # [x1, y1, x2, y2]

            if score > threshold:
                # Convert normalized coordinates to pixel coordinates
                x1, y1, x2, y2 = box
                # x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Choose color
                color = colors[i % len(colors)]
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw label with confidence score
                label_text = f"{label}: {score:.2f}"
                
                # Get text size for background
                bbox = draw.textbbox((x1, y1), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Draw background for text
                draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
                
                # Draw text
                draw.text((x1 + 2, y1 - text_height - 2), label_text, fill="white", font=font)
        
        return image_with_boxes

    def process_image(self, image, text_prompt, confidence_threshold=0.3, inference_format="Detection"):
        """
        Process image with Grounding DINO for object detection.
        
        Args:
            image: PIL Image
            text_prompt: Text description of objects to detect
            confidence_threshold: Minimum confidence for detections
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        Returns:
            Tuple of (image_with_boxes, detection_info, raw_results)
        """
        try:
            if image is None:
                return None, "Please upload an image.", ""
            
            
            if not text_prompt or text_prompt.strip() == "":
                return image, "Please enter a text prompt (e.g., 'a person. a car. a dog.')", ""
            
            result = self.model.detect_objects(
                    image=image,
                    text_prompt= text_prompt,
                    threshold=confidence_threshold,
                    )
            
            detections = result.get('detections', [])
            
            if len(detections) == 0:
                return image, f"No objects detected with confidence >= {confidence_threshold}", ""
            
            # Draw bounding boxes
            image_with_boxes = self.draw_bounding_boxes(
                image=image, 
                detections=detections, 
                threshold=confidence_threshold
            )
            
            # --- Classification Logic ---
            if inference_format == "Classification" and detections:
                # Filter to only the top-1 highest confidence detection
                top_detection = max(detections, key=lambda x: x['score'])
                detections = [top_detection]
                
                # Redraw boxes with only the top detection
                image_with_boxes = self.draw_bounding_boxes(
                    image=image, 
                    detections=detections, 
                    threshold=confidence_threshold
                )

            # --- Segmentation Logic ---
            if inference_format == "Segmentation" and detections:
                 try:
                     # 1. Prepare boxes -> [ [x1, y1, x2, y2], ... ]
                     bboxes = [det['box'] for det in detections]
                     
                     # 2. Run SAM
                     # SAM expects loaded image or path. PIL Image works.
                     # bboxes arg in ultralytics SAM: list of boxes
                     sam_results = self.sam_model(image, bboxes=bboxes, verbose=False)
                     
                     if sam_results and sam_results[0].masks:
                         # 3. Visualization
                         # Plot masks on top of image_with_boxes (or clean image?)
                         # Typically we want both boxes and masks. 
                         # plotting method from result returns a plotted numpy array
                         res_plotted = sam_results[0].plot() # numpy BGR
                         image_with_boxes = Image.fromarray(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
                         
                         # 4. Extract Polygons for return
                         # result.masks.xy is a list of arrays (one per mask)
                         masks_xy = sam_results[0].masks.xy
                         
                         # Update raw_results with segmentation
                         for i, det in enumerate(detections):
                             if i < len(masks_xy):
                                 # Convert numpy array to list of points [ [x,y], [x,y] ... ] or flattened?
                                 # COCO segmentation is usually [[x1, y1, x2, y2, ...]] (flattened)
                                 poly = masks_xy[i].flatten().tolist()
                                 det['segmentation'] = [poly]
                 except Exception as e:
                     print(f"SAM Error: {e}")
                     # Fallback to just boxes if SAM fails
                     pass

            # Prepare detection results for display

            # Prepare detection results for display
            detection_info = []
            for i, det in enumerate(detections):
                label = det['label']
                score = float(det['score'])
                x1, y1, x2, y2 = det['box']

                detection_info.append(f"Object {i+1}: {label} (confidence: {score:.3f})")
                detection_info.append(f"  Bounding box: ({x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f})")

            detection_text = "\n".join(detection_info)
            raw_results = detections
            
            return image_with_boxes, detection_text, raw_results
            
        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            print(error_msg)
            return image if image else None, error_msg, ""
        
    def inference_dataset(self, prompt='.',  confidence_threshold=0.3, inference_format="Detection"):
        
        sel_dataset = self.selected_dataset
        success, dataset_info = self.get_dataset_by_name(sel_dataset)
        
        if not success:
             return (
                f"<div style='padding: var(--size-2); border: 1px solid var(--block-border-color); "
                f"background: var(--input-background-fill); border-radius: var(--container-radius); "
                f"color: var(--body-text-color); min-height: 80px;'>"
                f"❌ Error: Dataset '{sel_dataset}' not found or has been cleaned up."
                f"</div>"
            )

        dataset_dir = Path(dataset_info.get('path', ''))

        output_dir = Path(self.datasets_dir).parent / '.output' / f"{self.selected_dataset}_coco" 
        output_dir.mkdir(parents=True, exist_ok=True)
        
        initial_categories = None 
        category_map = {}
        
        if inference_format == "Classification":
            initial_categories = []
        else:
            initial_categories = [{"id": 1, "name": "object", "supercategory": ""}]
            category_map["object"] = 1
            
        coco_builder = COCODatasetBuilder(
                        base_dir=output_dir.as_posix(),
                        contributor="Chee Yee",
                        description="Dataset",
                        version="1.0.0",
                        categories=initial_categories
                    )
        
        # Save inference format to metadata
        coco_builder.coco_json['info']['inference_format'] = inference_format
        
        imgs = [f for f in dataset_dir.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']]

        for img_path in imgs:
            img = Image.open(img_path).convert("RGB")
            width, height = img.size
            img_id = coco_builder.add_image(Path(img_path).name, width, height, verbose=False)

            annotated_image, text, detections = self.process_image(
                image= img,
                text_prompt= prompt,
                confidence_threshold= confidence_threshold,
                inference_format=inference_format
            )

            # Save Annotated Image
            save_dir = Path(coco_builder.directories['annotated_images'])/ Path(img_path).name
            annotated_image.save(save_dir)

            save_dir = Path(coco_builder.directories['train_images'])/ Path(img_path).name
            img.save(save_dir)

            for i, det in enumerate(detections):
                label = det['label']
                score = float(det['score'])
                xyxy = det['box']
                xywh = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])]


                # Determine Category ID
                cat_id = 1
                cat_name = "object"
                
                if inference_format == "Classification":
                    label_name = det.get('label', 'unknown')
                    
                    # Skip empty labels
                    if not label_name or label_name.strip() == '':
                        print(f"⚠️ Skipping annotation for empty label - image will have no tag")
                        continue
                    
                    # Skip annotations with combined labels (e.g., "bear cat")
                    if ' ' in label_name:
                        print(f"⚠️ Skipping annotation for combined label: '{label_name}' - image will have no tag")
                        continue
                    
                    if label_name in category_map:
                        cat_id = category_map[label_name]
                    else:
                        # Create new category on the fly
                        # Find next available ID (max of existing values or 0) + 1
                        existing_ids = category_map.values()
                        next_id = max(existing_ids) + 1 if existing_ids else 1
                        
                        coco_builder.add_category(next_id, label_name)
                        category_map[label_name] = next_id
                        cat_id = next_id
                        print(f"Created new category: {label_name} (ID: {cat_id})")

                coco_builder.add_annotation(
                    img_id=img_id,
                    category_id=cat_id,
                    xywh=xywh,
                    verbose=False,
                    segmentation=det.get('segmentation', [])
                )
        coco_builder.save_json(Path(coco_builder.directories['annotations'])/'instances_Train.json')
        shutil.make_archive(coco_builder.directories['base'], 'zip', coco_builder.directories['base'])

        return (
            f"<div style='padding: var(--size-2); border: 1px solid var(--block-border-color); "
            f"background: var(--input-background-fill); border-radius: var(--container-radius); "
            f"color: var(--body-text-color); min-height: 80px;'>"
            f"Inference completed on {len(imgs)} images.<br>Results saved to '<i>{output_dir}</i>'."
            f"</div>"
        )

    def _get_cvat_client(self):
        """Helper to create and return a configured CVAT client context manager."""
        url = self.config.get("cvat_url")
        username = self.config.get("cvat_username")
        password = self.config.get("cvat_password")

        if not all([url, username, password]):
            return None, "❌ Error: Missing CVAT credentials in settings.json."

        # Sanitize URL
        url = url.split('/projects')[0].split('/tasks')[0].split('jobs')[0].rstrip('/')

        try:
            configuration = Configuration(
                host=url,
                username=username,
                password=password,
            )
            return ApiClient(configuration), None
        except Exception as e:
            return None, f"❌ Configuration Error: {str(e)}"

    def get_cvat_projects(self):
        """Fetches list of projects from CVAT. Returns list of (name, id) tuples."""
        client, error = self._get_cvat_client()
        if error:
            print(error)
            return []

        try:
            with client:
                # projects_api.list() returns (data, response_headers) tuple usually
                # data is a list of ProjectRead objects
                projects, _ = client.projects_api.list()
                
                # Format as [(Name (ID: X), X)] for Gradio dropdown
                return [(f"{p.name} (ID: {p.id})", p.id) for p in projects.results if p.id==107]
        except Exception as e:
            print(f"Error fetching projects: {e}")
            return []

    def get_cvat_tasks(self,project_id=None):
        """Fetches list of tasks from CVAT. Returns list of (name, id) tuples."""
        client, error = self._get_cvat_client()
        if error:
            print(error)
            return []

        try:
            with client:
                # tasks_api.list() returns (data, response_headers) tuple usually
                # data is a list of TaskRead objects
                tasks, _ = client.tasks_api.list(project_id=project_id)
                
                # Format as [(Name (ID: X), X)] for Gradio dropdown
                return [(f"{t.name} (ID: {t.id})", t.id) for t in tasks.results]
        except Exception as e:
            print(f"Error fetching tasks: {e}")
            return []
    

    def create_cvat_task(self, project_id=None):
        """
        Create a CVAT task from the currently processed dataset using High-Level SDK.
        """
        url = self.config.get("cvat_url")
        username = self.config.get("cvat_username")
        password = self.config.get("cvat_password")
        
        if not all([url, username, password]):
            return "❌ Error: Missing CVAT credentials."
            
        # Sanitize URL
        url = url.split('/projects')[0].split('/tasks')[0].split('jobs')[0].rstrip('/')

        dataset_name = self.selected_dataset
        if not dataset_name:
            return "❌ Error: No dataset selected."

        # Locate the inference output directory
        output_dir = Path(self.datasets_dir).parent / '.output' / f"{dataset_name}_coco"
        images_dir = output_dir / "images" / "Train"
        annotations_file = output_dir / "annotations" / "instances_Train.json"

        if not images_dir.exists() or not annotations_file.exists():
            return f"❌ Error: Inference output not found at {output_dir}. Please run 'Inference Dataset' first."

        try:
            # Use High-Level 'make_client'
            with make_client(url, credentials=(username, password)) as client:
                
                # --- Label Synchronization (Added for Classification/Auto-Labeling) ---
                if project_id and annotations_file.exists():
                     try:
                         # Read categories from generated COCO file
                         with open(annotations_file, 'r') as f:
                             coco_data = json.load(f)
                         
                         dataset_categories = {cat['name'] for cat in coco_data.get('categories', [])}
                         
                         if dataset_categories:
                             print(f"Syncing categories to CVAT Project {project_id}: {dataset_categories}")
                             
                             # DEBUG: Check for 'dog' specifically
                             if "dog" in str(coco_data):
                                 print("⚠️ DEBUG: Found 'dog' in JSON file content!")
                             else:
                                 print("ℹ️ DEBUG: 'dog' NOT found in JSON file content.")
                             
                             # Retrieve project and existing labels
                             project_proxy = client.projects.retrieve(int(project_id))
                             current_labels = project_proxy.get_labels()
                             existing_names = {l.name for l in current_labels}
                             
                             new_labels = []
                             for cat_name in dataset_categories:
                                 if cat_name not in existing_names:
                                     print(f"  + Adding New Label to Project: {cat_name}")
                                     new_labels.append(
                                         models.PatchedLabelRequest(
                                             name=cat_name, 
                                             color="#ff0000", 
                                             attributes=[]
                                         )
                                     )
                             
                             if new_labels:
                                 # We must send ALL labels (existing + new) in a batch update
                                 # CRITICAL: Include 'id' for existing labels, otherwise CVAT tries to recreate them -> Unique Error
                                 updated_labels = [
                                     models.PatchedLabelRequest(
                                         id=l.id,
                                         name=l.name, 
                                         color=l.color, 
                                         attributes=[
                                             models.AttributeRequest(
                                                 name=a.name,
                                                 input_type=a.input_type,
                                                 mutable=a.mutable,
                                                 values=a.values
                                             ) for a in l.attributes
                                         ]
                                     ) for l in current_labels
                                 ]
                                 updated_labels.extend(new_labels)
                                 
                                 project_proxy.update(models.PatchedProjectWriteRequest(labels=updated_labels))
                                 print(f"✅ Successfully synced {len(new_labels)} new labels to project.")
                     except Exception as e:
                         print(f"⚠️ Warning: Label synchronization failed (Non-critical): {e}")

                # Define task spec first
                task_spec = {
                    "name": f"{dataset_name}_vlm_annotation",
                    "project_id": int(project_id) if project_id else None,
                }

                # Fetch project to get organization context if needed
                if project_id:
                    try:
                        project = client.projects.retrieve(int(project_id))
                        # project.organization usually returns the Organization ID (int)
                        org_id = project.organization
                        
                        if org_id:
                            print(f"Project belongs to Organization ID: {org_id}")
                            # Fetch the exact slug which is needed for the context header
                            org = client.organizations.retrieve(org_id)
                            org_slug = org.slug
                            print(f"Setting Client Context to Organization: {org_slug}")
                            
                            # Set the context for the client
                            client.organization_slug = org_slug
                            
                            # Also update the spec with the slug/name just in case
                            task_spec["organization"] = org_slug
                            
                    except Exception as e:
                        print(f"Warning: Could not fetch project/organization details: {e}")

                print(f"Creating CVAT task (High-Level): {task_spec['name']}...")
                task = client.tasks.create(spec=task_spec)
                
                # Upload Data
                print("Uploading images...")
                image_files = [str(f) for f in images_dir.iterdir() if f.is_file()]
                task.upload_data(image_files)

                # Upload Annotations
                print("Uploading annotations...")
                task.import_annotations(
                    format_name="COCO 1.0",
                    filename=str(annotations_file)
                )
                
                # --- Tag Annotation Logic (LabeledImage) ---
                # Apply Classification tags to the frames
                if annotations_file.exists():
                     try:
                         if 'coco_data' not in locals():
                             with open(annotations_file, 'r') as f:
                                 coco_data = json.load(f)
                                 
                         # Check Inference Format from Metadata
                         inf_format = coco_data.get('info', {}).get('inference_format', 'Detection')
                         print(f"DEBUG: Dataset Inference Format: {inf_format}")
                         
                         if inf_format == "Classification":
                             print("Applying Tag Annotations (Classification)...")
                             
                             # 1. Retrieve Task Labels to get internal IDs
                             # task = client.tasks.retrieve(task.id) # 'task' object is already returned by create
                             task_labels = task.get_labels()
                             label_name_to_id = {l.name: l.id for l in task_labels}
                             
                             # 2. Prepare mapping of Image Filename -> Frame Index
                             if 'coco_data' not in locals():
                                 with open(annotations_file, 'r') as f:
                                     coco_data = json.load(f)
                             
                             images_list = coco_data.get('images', [])
                             # Sort by file_name to match CVAT order
                             sorted_images = sorted(images_list, key=lambda x: x['file_name'])
                             
                             image_id_to_frame = {}
                             for idx, img_info in enumerate(sorted_images):
                                 image_id_to_frame[img_info['id']] = idx
                                 
                             # 3. Create Tags from Annotations
                             annotations = coco_data.get('annotations', [])
                             categories = {c['id']: c['name'] for c in coco_data.get('categories', [])}
                             
                             tags_to_create = []
                             
                             for ann in annotations:
                                 img_id = ann['image_id']
                                 cat_id = ann['category_id']
                                 
                                 if img_id not in image_id_to_frame: continue
                                 
                                 frame_idx = image_id_to_frame[img_id]
                                 cat_name = categories.get(cat_id)
                                 
                                 # Skip empty category names
                                 if not cat_name or cat_name.strip() == '':
                                     print(f"⚠️ Skipping CVAT tag for empty category name")
                                     continue
                                 
                                 # Skip combined labels (e.g., "bear cat") for classification
                                 if cat_name and ' ' in cat_name:
                                     print(f"⚠️ Skipping CVAT tag for combined label: '{cat_name}'")
                                     continue
                                 
                                 if cat_name and cat_name in label_name_to_id:
                                     cvat_label_id = int(label_name_to_id[cat_name])
                                     
                                     # Simplify Request: Omit attributes/group if default
                                     tag_annotation = models.LabeledImageRequest(
                                         frame=int(frame_idx),
                                         label_id=cvat_label_id
                                     )
                                     tags_to_create.append(tag_annotation)
                             
                             if tags_to_create:
                                 # 4. Upload Tags to Job 0
                                 jobs = task.get_jobs()
                                 if jobs:
                                     target_job = jobs[0]
                                     
                                     patch_request = models.PatchedLabeledDataRequest(
                                         tags=tags_to_create
                                     )
                                     
                                     
                                     target_job.update_annotations(patch_request, action=SimpleNamespace(value="create"))
                                     print(f"✅ Created {len(tags_to_create)} Tag Annotations on Job {target_job.id}.")
                                 else:
                                     print("⚠️ Warning: No jobs found for task. Skipping tags.")
                             else:
                                 print("ℹ️ No tags to create.")
    
                         else:
                             print(f"ℹ️ Skipping Tag Annotation for format: {inf_format}")
                             
                     except Exception as e:
                         print(f"⚠️ Warning: Tag Annotation failed: {e}")
                         traceback.print_exc()
                         
                print(f"Task {task.name} (ID: {task.id}) created successfully.")
                
                task_url = f"{url.rstrip('/')}/tasks/{task.id}"
                
                # Cleanup output directory after successful upload
                shutil.rmtree(output_dir, ignore_errors=True)
                
                # Also remove the generated zip file
                zip_path = output_dir.with_suffix(".zip")
                if zip_path.exists():
                    zip_path.unlink()
                
                return (
                    f"<div style='padding: var(--size-2); border: 1px solid var(--block-border-color); "
                    f"background: var(--input-background-fill); border-radius: var(--container-radius); "
                    f"color: var(--body-text-color); min-height: 80px;'>"
                    f"<strong>✅ CVAT Task Created Successfully! (High-Level)</strong><br><br>"
                    f"<b>ID:</b> {task.id}<br>"
                    f"<b>Name:</b> {task.name}<br>"
                    f"<b>URL:</b> <a href='{task_url}' target='_blank' style='color: var(--link-text-color); text-decoration: underline;'>Open Task</a>"
                    f"</div>"
                )

        except Exception as e:
            return f"❌ Error creating CVAT task: {str(e)}"
    
            # Cleanup on fail
            if 'temp_extract_dir' in locals(): shutil.rmtree(temp_extract_dir, ignore_errors=True)
            if 'temp_build_dir' in locals(): shutil.rmtree(temp_build_dir, ignore_errors=True)
            
            print(f"Detailed Error: {e}")
            return f"❌ Error during formatting: {str(e)}", None

    def _download_and_format_task(self, task_id, output_zip_path, split_ratios=(70, 20, 10), format_name="Ultralytics YOLO Detection 1.0"):
        """
        Helper method to download and format a task to a specific location.
        Returns (SuccessBool, Message)
        split_ratios: tuple of (train, val, test) percentages. Should sum roughly to 100.
        """
        output_zip_path = Path(output_zip_path)
        print(f"DEBUG: Internal Processing CVAT Task {task_id} -> {output_zip_path}")
        
        url = self.config.get("cvat_url")
        username = self.config.get("cvat_username")
        password = self.config.get("cvat_password")
        
        # --- PHASE 1: DOWNLOAD ---
        host = url.split('/projects')[0].split('/tasks')[0].split('jobs')[0].rstrip('/')
        raw_zip_path = Path(self.datasets_dir) / f"cvat_task_{task_id}.zip"
        
        try:
            print(f"Connecting to CVAT to download Task {task_id}...")
            with make_client(host, credentials=(username, password)) as client:
                task = client.tasks.retrieve(int(task_id))
                raw_zip_path.parent.mkdir(parents=True, exist_ok=True)
                if raw_zip_path.exists():
                    raw_zip_path.unlink()
                
                task.export_dataset(
                    format_name=format_name,
                    filename=str(raw_zip_path),
                    include_images=True
                )
        except Exception as e:
            return False, f"Error downloading: {str(e)}"

        # --- PHASE 2: FORMAT ---
        temp_extract_dir = Path(self.datasets_dir) / f"temp_extract_{task_id}_{int(time.time())}"
        temp_build_dir = Path(self.datasets_dir) / f"temp_build_{task_id}_{int(time.time())}"
        
        try:
            with zipfile.ZipFile(raw_zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)
            
            # Detect format type
            is_classification = "Classification" in format_name
            
            if is_classification:
                # --- CLASSIFICATION FORMAT ---
                # Structure: temp_extract_dir/train/dog/, train/cat/, etc.
                src_train_dir = temp_extract_dir / "train"
                
                if not src_train_dir.exists():
                    return False, "Error: 'train' directory not found for Classification format."
                
                # Get all class folders
                class_folders = [d for d in src_train_dir.iterdir() if d.is_dir()]
                
                if not class_folders:
                    return False, "Error: No class folders found in 'train' directory."
                
                print(f"Found {len(class_folders)} classes: {[d.name for d in class_folders]}")
                
                # Normalize split ratios
                r_train, r_val, r_test = split_ratios
                total_r = r_train + r_val + r_test
                if total_r == 0: total_r = 100
                
                # Create output structure for each class
                for class_folder in class_folders:
                    class_name = class_folder.name
                    
                    # Create directories for this class
                    (temp_build_dir / "train" / class_name).mkdir(parents=True, exist_ok=True)
                    (temp_build_dir / "val" / class_name).mkdir(parents=True, exist_ok=True)
                    (temp_build_dir / "test" / class_name).mkdir(parents=True, exist_ok=True)
                    
                    # Get all images in this class folder
                    images = [f for f in class_folder.iterdir() 
                             if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                    
                    if not images:
                        print(f"Warning: No images found in class '{class_name}'")
                        continue
                    
                    # Shuffle and split
                    random.seed(42)
                    random.shuffle(images)
                    n_total = len(images)
                    
                    n_train = int(n_total * (r_train / total_r))
                    n_val = int(n_total * (r_val / total_r))
                    n_test = n_total - n_train - n_val
                    
                    train_imgs = images[:n_train]
                    val_imgs = images[n_train:n_train+n_val]
                    test_imgs = images[n_train+n_val:]
                    
                    # Copy images to respective folders
                    for img in train_imgs:
                        shutil.copy2(img, temp_build_dir / "train" / class_name / img.name)
                    for img in val_imgs:
                        shutil.copy2(img, temp_build_dir / "val" / class_name / img.name)
                    for img in test_imgs:
                        shutil.copy2(img, temp_build_dir / "test" / class_name / img.name)
                    
                    print(f"Class '{class_name}': {n_train} train, {n_val} val, {n_test} test")
                
                # No data.yaml needed for Classification - YOLOv8 infers classes from folder structure
                
            else:
                # --- DETECTION/SEGMENTATION FORMAT ---
                # Structure: temp_extract_dir/images/train/, labels/train/
                src_images_dir = temp_extract_dir / "images" / "train"
                src_labels_dir = temp_extract_dir / "labels" / "train"
                
                if not src_images_dir.exists():
                    return False, "Error: 'images/train' not found."

                images = [f for f in src_images_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                
                # Shuffle and Split based on ratios
                random.seed(42)
                random.shuffle(images)
                n_total = len(images)
                
                # Normalize ratios to ensure they sum to 100 (or handle raw counts, but percents are easier)
                r_train, r_val, r_test = split_ratios
                total_r = r_train + r_val + r_test
                if total_r == 0: total_r = 100 # Avoid div by zero
                
                n_train = int(n_total * (r_train / total_r))
                n_val = int(n_total * (r_val / total_r))
                # Assign remaining to test to ensure all images are used (and handle rounding errors)
                n_test = n_total - n_train - n_val
                
                train_images = images[:n_train]
                val_images = images[n_train:n_train+n_val]
                test_images = images[n_train+n_val:]
                
                # Prepare Structure
                dirs_to_create = [
                    temp_build_dir / "images" / "Train",
                    temp_build_dir / "images" / "Validation",
                    temp_build_dir / "images" / "Test",
                    temp_build_dir / "labels" / "Train",
                    temp_build_dir / "labels" / "Validation",
                    temp_build_dir / "labels" / "Test"
                ]
                for d in dirs_to_create:
                    d.mkdir(parents=True, exist_ok=True)
                
                train_txt_lines = []
                val_txt_lines = []
                test_txt_lines = []
                
                def process_files(file_list, dest_subset, txt_list):
                    for img_path in file_list:
                        shutil.copy2(img_path, temp_build_dir / "images" / dest_subset / img_path.name)
                        label_name = img_path.stem + ".txt"
                        src_label = src_labels_dir / label_name
                        if src_label.exists():
                            shutil.copy2(src_label, temp_build_dir / "labels" / dest_subset / label_name)
                        txt_list.append(f"./images/{dest_subset}/{img_path.name}")

                process_files(train_images, "Train", train_txt_lines)
                process_files(val_images, "Validation", val_txt_lines)
                process_files(test_images, "Test", test_txt_lines)
                
                with open(temp_build_dir / "Train.txt", "w") as f: f.write("\n".join(train_txt_lines))
                with open(temp_build_dir / "Validation.txt", "w") as f: f.write("\n".join(val_txt_lines))
                with open(temp_build_dir / "Test.txt", "w") as f: f.write("\n".join(test_txt_lines))
                    
                # Modify data.yaml
                orig_yaml = temp_extract_dir / "data.yaml"
                if orig_yaml.exists():
                    with open(orig_yaml, 'r') as f:
                        yaml_content = f.read()
                    filtered_lines = [l for l in yaml_content.splitlines() if not (l.strip().startswith('train:') or l.strip().startswith('val:') or l.strip().startswith('validation:') or l.strip().startswith('test:'))]
                    final_yaml_content = "train: Train.txt\nval: Validation.txt\ntest: Test.txt\n" + "\n".join(filtered_lines)
                    with open(temp_build_dir / "data.yaml", "w") as f:
                        f.write(final_yaml_content)
            
            # Zip
            output_zip_path.parent.mkdir(parents=True, exist_ok=True)
            if output_zip_path.exists():
                output_zip_path.unlink()
            shutil.make_archive(
                base_name=str(output_zip_path).replace('.zip', ''),
                format='zip',
                root_dir=temp_build_dir
            )
            
            # Cleanup Raw
            if raw_zip_path.exists(): raw_zip_path.unlink()
            
            return True, "Success"
            
        except Exception as e:
            return False, f"Format Error: {str(e)}"
        finally:
            if temp_extract_dir.exists(): shutil.rmtree(temp_extract_dir, ignore_errors=True)
            if temp_build_dir.exists(): shutil.rmtree(temp_build_dir, ignore_errors=True)


    def process_cvat_task(self, task_id, custom_name=None, split_ratios=(70, 20, 10), format_name="Ultralytics YOLO Detection 1.0"):
        """
        Public wrapper to download/format and register the dataset.
        """
        if not task_id: return "❌ Error: No task selected.", None

        # Determine output filename
        if custom_name and custom_name.strip():
            safe_name = custom_name.strip()
            if not safe_name.lower().endswith('.zip'): safe_name += '.zip'
            formatted_zip_path = Path(self.datasets_dir) / safe_name
            dataset_entry_name = safe_name.replace('.zip', '')
        else:
            formatted_zip_path = Path(self.datasets_dir) / f"formatted_task_{task_id}.zip"
            dataset_entry_name = f"Task_{task_id}"
            
        success, msg = self._download_and_format_task(task_id, formatted_zip_path, split_ratios, format_name=format_name)
        
        if not success:
            return f"❌ {msg}", None
            
        # Register in Config
        if "formatted_datasets" not in self.config:
            self.config["formatted_datasets"] = []
        
        # Check redundancy/Update
        existing = next((item for item in self.config["formatted_datasets"] if item["name"] == dataset_entry_name), None)
        if not existing:
             self.config["formatted_datasets"].append({"name": dataset_entry_name, "path": str(formatted_zip_path)})
             self.save_config()
             
        return f"✅ Downloaded & Formatted Successfully!\\nPath: {formatted_zip_path}", str(formatted_zip_path)

    def save_formatted_dataset(self, zip_file_path):
        """
        Save uploaded formatted dataset to .gradio/own_formatted/ and update settings.json
        
        Args:
            zip_file_path: Path to uploaded formatted dataset zip
            
        Returns:
            tuple: (success: bool, message: str, dataset_name: str)
        """
        if not zip_file_path:
            return False, "❌ No file uploaded", None
        
        try:
            uploaded_path = Path(zip_file_path)
            if not uploaded_path.exists():
                return False, f"❌ File not found: {zip_file_path}", None
            
            # Create own_formatted directory
            own_formatted_dir = Path(self.datasets_dir) / "own_formatted"
            own_formatted_dir.mkdir(parents=True, exist_ok=True)
            
            # Use original filename (without .zip for dataset name)
            dataset_name = uploaded_path.stem  # Remove .zip extension
            dest_path = own_formatted_dir / uploaded_path.name
            
            # Copy file to destination
            shutil.copy2(uploaded_path, dest_path)
            
            print(f"📦 Saved formatted dataset: {uploaded_path.name} → {dest_path}")
            
            # Update settings.json - save to own_datasets (user-uploaded)
            if "own_datasets" not in self.config:
                self.config["own_datasets"] = []
            
            # Check if dataset already exists
            existing_idx = next((i for i, d in enumerate(self.config["own_datasets"]) 
                               if d.get("name") == dataset_name), -1)
            
            dataset_entry = {
                "name": dataset_name,
                "path": str(dest_path)
            }
            
            if existing_idx >= 0:
                # Update existing path
                self.config["own_datasets"][existing_idx] = dataset_entry
                message = f"✅ Formatted dataset updated: {dataset_name}"
            else:
                # Add new entry
                self.config["own_datasets"].append(dataset_entry)
                message = f"✅ Formatted dataset saved: {dataset_name}"
            
            self.save_config()
            
            return True, message, dataset_name
            
        except Exception as e:
            print(f"Error saving formatted dataset: {e}")
            import traceback
            traceback.print_exc()
            return False, f"❌ Error: {str(e)}", None
    
    def delete_formatted_dataset(self, dataset_name):
        """
        Delete formatted dataset from filesystem and settings.json
        
        Args:
            dataset_name: Name of formatted dataset to delete
            
        Returns:
            tuple: (success: bool, message: str)
        """
        if not dataset_name:
            return False, "❌ Please select a dataset to delete"
        
        try:
            own_datasets = self.config.get("own_datasets", [])
            dataset_entry = next((d for d in own_datasets if d.get("name") == dataset_name), None)
            
            if not dataset_entry:
                return False, f"❌ Dataset '{dataset_name}' not found in settings"
            
            # Delete physical file
            dataset_path = Path(dataset_entry.get("path", ""))
            if dataset_path.exists():
                os.remove(dataset_path)
                print(f"🗑️ Deleted file: {dataset_path}")
            else:
                print(f"⚠️ File not found (already deleted?): {dataset_path}")
            
            # Remove from config
            self.config["own_datasets"] = [
                d for d in own_datasets if d.get("name") != dataset_name
            ]
            self.save_config()
            
            return True, f"✅ Formatted dataset '{dataset_name}' deleted successfully"
            
        except Exception as e:
            print(f"Error deleting formatted dataset: {e}")
            import traceback
            traceback.print_exc()
            return False, f"❌ Error: {str(e)}"


    def cleanup_preview(self):
        """Clean up all temporary preview files"""
        try:
             # Find all regular preview zips
             for p in Path(self.datasets_dir).glob("temp_preview_task_*.zip"):
                 p.unlink()
             # Find all temp inspect folders
             for p in Path(self.datasets_dir).glob("temp_inspect_*"):
                 if p.is_dir(): shutil.rmtree(p)
        except Exception as e:
            print(f"Cleanup warning: {e}")

    def _sanitize_stats_for_json(self, data):
        """Recursively ensure all dictionary keys are strings for JSON compatibility."""
        if isinstance(data, dict):
            return {str(k): self._sanitize_stats_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_stats_for_json(i) for i in data]
        else:
            return data

    def inspect_dataset_zip(self, zip_path):
        """
        Inspect a local dataset zip file and return stats.
        Supports both Detection/Segmentation and Classification formats.
        """
        if not zip_path: return {"status": "Error", "message": "No path provided"}
        
        zip_path = Path(zip_path)
        if not zip_path.exists():
             return {"status": "Error", "message": f"File not found: {zip_path}"}
             
        # Extract to temp inspect folder
        import time
        import shutil
        import zipfile
        
        temp_inspect_dir = Path(self.datasets_dir) / f"temp_inspect_{int(time.time())}_{random.randint(1000,9999)}"
        
        try:
            temp_inspect_dir.mkdir(parents=True, exist_ok=True)
            
            # Use smart extraction that auto-flattens nested structures
            self.extract_and_flatten_zip(zip_path, temp_inspect_dir)
                
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
                        import yaml
                        data = yaml.safe_load(f)
                        stats["classes"] = data.get('nc', 0)
                        stats["class_names"] = data.get('names', [])
            
            # Apply robust sanitization before returning
            return self._sanitize_stats_for_json(stats)
            
        except Exception as e:
            print(f"Error inspecting zip: {e}")
            return {"status": "Error", "message": str(e)}
        finally:
            if temp_inspect_dir.exists(): shutil.rmtree(temp_inspect_dir)

if __name__ == "__main__":
    # Load model into device: GPU or CPU
    print('Loading the model and processor...')
    model_id = "IDEA-Research/grounding-dino-base"
    vlm_model = GroundingDINODetector(model_id=model_id)

    dataset_builder = COCODatasetBuilder()

    result = vlm_model.process_single_image(
        image_path= "./Gitlab/SEEAI/.output/globe/27_4_72-MW_1.jpg",
        text_prompt="Yellow Wire attached to a ball bond.",
        threshold=0.1,
        save_result=False,
        show_plot=False,
        verbose=False
    )

    result = vlm_model.detect_objects(
        image= "./Gitlab/SEEAI/.output/globe/27_4_72-MW_1.jpg",
        text_prompt="Yellow Wire attached to a ball bond.",
        threshold=0.1,
    )

    img_path = './Gitlab/.asset/HE_9_7_O.jpg'
    image = Image.open(img_path).convert("RGB")
    text_prompt = "Car."
    confidence_threshold = 0.3
    # process_result = gdino.process_image(image, text_prompt, confidence_threshold)
    # print(process_result)

    a = APP(vlm_model=vlm_model)
    processed_result = a.process_image(image=image, text_prompt=text_prompt, confidence_threshold=confidence_threshold)

    a.select_dataset('globe')
    a.inference_dataset(prompt='.', confidence_threshold=0.1)
    # a.select_dataset('globe')
    # sel_dataset = a.selected_dataset
    # success, dataset = a.get_dataset_by_name(sel_dataset)
    # dataset_name, dataset_path = dataset.get('name', ''), dataset.get('path', '')
    # dataset_path = Path(dataset_path)

    # builder = COCODatasetBuilder(
    #     base_dir='output/coco2',
    #     contributor="Chee Yee",
    #     description="Dataset",
    #     version="1.0.0",
    #     categories=[
    #                 {"id": 1, "name": "object", "supercategory": ""},
    #             ]
    # )
    # imgs = [f for f in dataset_path.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']]
    # for img_path in imgs:
    #     img = Image.open(img_path).convert("RGB")
    #     width, height = img.size
    #     img_id = builder.add_image(Path(img_path).name, width, height, verbose=False)

    #     annotated_image, text, detections = a.process_image(
    #         image= img,
    #         text_prompt="Yellow Wire attached to a ball bond.",
    #         confidence_threshold=0.1
    #     )

    #     # Save Annotated Image
    #     save_dir = Path(builder.directories['annotated_images'])/ Path(img_path).name
    #     annotated_image.save(save_dir)

    #     save_dir = Path(builder.directories['train_images'])/ Path(img_path).name
    #     img.save(save_dir)

    #     for i, det in enumerate(detections):
    #         label = det['label']
    #         score = float(det['score'])
    #         xyxy = det['box']
    #         xywh = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])]

    #         builder.add_annotation(
    #             img_id=img_id,
    #             category_id=1,
    #             xywh=xywh,
    #             verbose=False
    #         )
    # builder.save_json(Path(builder.directories['annotations'])/'instances_Train.json')
    # shutil.make_archive(builder.directories['base'], 'zip', 'output/coco2')



