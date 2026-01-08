import traceback
import json
import time
import os
import shutil
import zipfile
import random
import io
from pathlib import Path
import tempfile
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
from ModelManager.DBmanager import DBManager

from .utils import COCODatasetBuilder, GroundingDINODetector
from src.ModelManager.utils import extract_and_flatten_zip, upload_dataset_by_zip
from ultralytics import YOLO, SAM
from src.ModelManager.config import CVAT_HOST_IP, CVAT_HOST_PORT, CVAT_USER, CVAT_PASSWORD
CVAT_HOST = str(CVAT_HOST_IP) + ":" + str(CVAT_HOST_PORT)

class APP():

    def __init__(self, vlm_model: GroundingDINODetector=None):
        # In-memory dataset storage (dict format: {name: {name, path}})
        self.datasets = {}
        self.datasets_dir = str(Path(__file__).parent.parent / "ModelManager/datasets/temp")
        self.selected_dataset = ''
        self.selected_dataset_1st_img_path = ""

        self.sam_model = SAM("sam2.1_b.pt")
        self.model = vlm_model
    
    def get_all_datasets(self, name=None):
        # Returns a list of datasets, each element is a dict with 'name' and 'path'
        return list(self.datasets.values())

    def get_dataset_by_name(self, name):
        """
        Retrieves a single dataset's details by its name.
        """
        dataset = self.datasets.get(name)
        if dataset:
            return True, dataset
        return False, f"Dataset '{name}' not found."    
    
    def remove_dataset(self, name):
        """
        Removes a dataset from memory and deletes its directory from the file system.
        """
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
            message = f"Directory for '{name}' not found on disk, only removing from memory."
        
        # Also remove associated zip file if it exists
        if dataset_path:
            zip_file_path = f"{dataset_path}.zip"
            if os.path.exists(zip_file_path):
                try:
                    os.remove(zip_file_path)
                    print(f"🗑️  Deleted associated zip file: {zip_file_path}")
                except OSError as e:
                    print(f"⚠️  Warning: Could not delete zip file {zip_file_path}: {e}")
            
        # Remove the dataset from memory
        del self.datasets[name]

        return True, f"Dataset '{name}' removed successfully."


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
    #                         ZIPFILE LOGIC
    # -------------------------------------------------------------------------
    def extract_and_flatten_zip(self, zip_path, extract_to):
        """
        Wrapper method that calls the utility function.
        Kept for backward compatibility.
        """
        return extract_and_flatten_zip(zip_path, extract_to)

    def upload_dataset_by_zip(self, zip_file_path):
        """
        Uploads a dataset from zip file, extracts it, and stores in memory.
        """
        success, message, dataset_info = upload_dataset_by_zip(
            zip_file_path,
            self.datasets_dir
        )
        
        if success and dataset_info:
            # Add to in-memory datasets dictionary
            self.datasets[dataset_info["name"]] = dataset_info
        
        return success, message, dataset_info.get("name", "") if dataset_info else ""
    # -------------------------------------------------------------------------
    #                         PREDICT MODEL LOGIC
    # -------------------------------------------------------------------------

    def get_model_plots(self, model_path):
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
                
        # Return list of (label, path) tuples for Gallery, or just paths?
        # Gradio Gallery accepts list of (path, label) tuples.
        return [(path, label) for label, path in plots]

    # -------------------------------------------------------------------------
    #                         TRAINING LOGIC
    # -------------------------------------------------------------------------
    def get_format_pretrained_models(self, format_name=None):
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
        
        models_dir = Path(f"src/ModelManager/models/pre_trained/{sub_dir}")
        if not models_dir.exists():
            return []
        
        return [f.name for f in models_dir.glob("*.pt")]

    def get_random_sample_images(self, project_id, count=4):
        """
        Extracts up to 'count' random images from the CVAT project zip.
        Downloads the zip if it doesn't exist.
        Returns: List of PIL Image objects
        """
        if not project_id:
            return None
        
        # 1. Check/Download Zip
        raw_zip_path = Path(self.datasets_dir) / f"cvat_project_{project_id}.zip"
        
        if not raw_zip_path.exists():
            # Attempt download (simplified version of download logic)
            try:
                url = CVAT_HOST
                username = CVAT_USERNAME
                password = CVAT_PASSWORD
                host = url.split('/projects')[0].split('/tasks')[0].split('jobs')[0].rstrip('/')
                
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
        
        path_obj = Path(zip_path)
        if not path_obj.exists():
             return {"status": "Error", "message": f"Path not found: {zip_path}"}
             
        # If it's already a directory, inspect it directly
        if path_obj.is_dir():
            temp_inspect_dir = path_obj
            is_temp = False
        else:
            # Extract to temp inspect folder
            import time
            import shutil
            import zipfile
            
            temp_inspect_dir = Path(self.datasets_dir) / f"temp_inspect_{int(time.time())}_{random.randint(1000,9999)}"
            temp_inspect_dir.mkdir(parents=True, exist_ok=True)
            is_temp = True
            
            try:
                # Use smart extraction that auto-flattens nested structures
                self.extract_and_flatten_zip(zip_path, temp_inspect_dir)
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
            if 'is_temp' in locals() and is_temp and temp_inspect_dir.exists(): 
                shutil.rmtree(temp_inspect_dir)
    # -------------------------------------------------------------------------
    #                         EXTRACT FRAMES FROM VIDEO LOGIC
    # -------------------------------------------------------------------------
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

        # 4. Zip the new dataset so it can be re-uploaded or used as main source
        # We need to zip the contents of temp_dir
        zip_output_path = os.path.join(self.datasets_dir, f"{temp_name}.zip")
        with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
                    
        # Update in-memory datasets with new temp dataset
        new_entry = {"name": temp_name, "path": temp_dir}
        self.datasets[temp_name] = new_entry
        
        self.select_dataset(temp_name)

        return True, zip_output_path
    
    # -------------------------------------------------------------------------
    #                         INFERENCE DATASET LOGIC
    # -------------------------------------------------------------------------
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
            
            # Split prompt by common delimiters (comma or period) to avoid combined labels
            # Example: "bear, bird, horse" -> ["bear", "bird", "horse"]
            import re
            # Split by comma or period, then clean up whitespace
            prompt_parts = re.split(r'[,.]', text_prompt)
            prompt_parts = [p.strip() for p in prompt_parts if p.strip()]
            
            # If only one prompt or empty, use original behavior
            if len(prompt_parts) <= 1:
                result = self.model.detect_objects(
                    image=image,
                    text_prompt=text_prompt,
                    threshold=confidence_threshold,
                )
                detections = result.get('detections', [])
            else:
                # Run detection separately for each class to avoid combined labels
                print(f"🔀 Splitting prompt into {len(prompt_parts)} parts: {prompt_parts}")
                all_detections = []
                
                for single_prompt in prompt_parts:
                    # Add period to help GroundingDINO distinguish separate entities
                    prompt_with_period = single_prompt if single_prompt.endswith('.') else f"{single_prompt}."
                    
                    result = self.model.detect_objects(
                        image=image,
                        text_prompt=prompt_with_period,
                        threshold=confidence_threshold,
                    )
                    single_detections = result.get('detections', [])
                    all_detections.extend(single_detections)
                    print(f"  ✓ Detected {len(single_detections)} objects for '{single_prompt}'")
                
                # Apply NMS: Group overlapping boxes and keep highest confidence per group
                if all_detections:
                    def calculate_iou(box1, box2):
                        """Calculate Intersection over Union between two boxes"""
                        x1_min, y1_min, x1_max, y1_max = box1
                        x2_min, y2_min, x2_max, y2_max = box2
                        
                        # Calculate intersection
                        inter_x_min = max(x1_min, x2_min)
                        inter_y_min = max(y1_min, y2_min)
                        inter_x_max = min(x1_max, x2_max)
                        inter_y_max = min(y1_max, y2_max)
                        
                        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
                            return 0.0
                        
                        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
                        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
                        union_area = box1_area + box2_area - inter_area
                        
                        return inter_area / union_area if union_area > 0 else 0.0
                    
                    # Group overlapping detections using NMS
                    iou_threshold = 0.5  # Boxes with IoU > 0.5 are considered same object
                    sorted_detections = sorted(all_detections, key=lambda x: x['score'], reverse=True)
                    final_detections = []
                    
                    while sorted_detections:
                        # Take the highest confidence detection
                        best_det = sorted_detections.pop(0)
                        final_detections.append(best_det)
                        
                        # Remove all detections that overlap significantly with this one
                        remaining = []
                        for det in sorted_detections:
                            iou = calculate_iou(best_det['box'], det['box'])
                            if iou <= iou_threshold:
                                # Keep detections that don't overlap much
                                remaining.append(det)
                            # else: discard overlapping lower-confidence detection
                        
                        sorted_detections = remaining
                    
                    detections = final_detections
                    print(f"📊 NMS: Kept {len(detections)} detections after removing overlaps")
                    for det in detections:
                        print(f"  ✓ {det['label']} (conf: {det['score']:.3f})")
                else:
                    detections = []
                    print(f"📊 No detections found")
            
                
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

        output_dir = Path(self.datasets_dir)/ '.output' / f"{self.selected_dataset}_coco" 
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Start with empty categories for all formats
        # Categories will be created dynamically based on detected labels
        initial_categories = []
        category_map = {}
            
        coco_builder = COCODatasetBuilder(
                        base_dir=output_dir.as_posix(),
                        contributor="Chee Yee",
                        description="Dataset",
                        version="1.0.0",
                        categories=initial_categories
                    )
        
        # Save inference format to metadata
        coco_builder.coco_json['info']['inference_format'] = inference_format
        
        imgs = [f for f in dataset_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']]

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


                # Determine Category ID - Dynamic for all formats
                label_name = det.get('label', 'unknown')
                
                # Skip empty labels
                if not label_name or label_name.strip() == '':
                    print(f"⚠️ Skipping annotation for empty label")
                    continue
                # Check if category already exists4
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
                    print(f"✨ Created new category: {label_name} (ID: {cat_id})")

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
    # -------------------------------------------------------------------------
    #                         CVAT LOGIC
    # -------------------------------------------------------------------------
    
    def create_cvat_project_with_tasks(self):
        """
        Create a CVAT project and split dataset into Train/Val/Test tasks (7:1:2 ratio).
        """
        url = CVAT_HOST
        username = CVAT_USER
        password = CVAT_PASSWORD
        org_id=1
        
        if not all([url, username, password]):
            return "❌ Error: Missing CVAT credentials."
            
        # Sanitize URL
        url = url.split('/projects')[0].split('/tasks')[0].split('jobs')[0].rstrip('/')

        dataset_name = self.selected_dataset
        if not dataset_name:
            return "❌ Error: No dataset selected."

        # Locate the inference output directory
        output_dir = Path(self.datasets_dir)/ '.output' / f"{dataset_name}_coco"
        images_dir = output_dir / "images" / "Train"
        annotations_file = output_dir / "annotations" / "instances_Train.json"

        if not images_dir.exists() or not annotations_file.exists():
            return f"❌ Error: Inference output not found at {output_dir}. Please run 'Inference Dataset' first."

        try:
            # Use High-Level 'make_client'
            with make_client(url, credentials=(username, password)) as client:
                
                # 1. Load COCO data
                print("📖 Loading COCO annotations...")
                with open(annotations_file, 'r') as f:
                    coco_data = json.load(f)
                
                images = coco_data.get('images', [])
                annotations = coco_data.get('annotations', [])
                categories = coco_data.get('categories', [])
                inf_format = coco_data.get('info', {}).get('inference_format', 'Detection')
                
                # 2. Split dataset (7:1:2 ratio)
                print(f"🔀 Splitting {len(images)} images into Train/Val/Test (7:1:2)...")
                import random
                random.seed(42)
                shuffled_images = random.sample(images, len(images))
                
                n_total = len(shuffled_images)
                n_train = int(n_total * 0.7)
                n_val = int(n_total * 0.1)
                
                train_images = shuffled_images[:n_train]
                val_images = shuffled_images[n_train:n_train+n_val]
                test_images = shuffled_images[n_train+n_val:]
                
                print(f"  📊 Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
                
                # Create image_id sets for filtering annotations
                train_ids = {img['id'] for img in train_images}
                val_ids = {img['id'] for img in val_images}
                test_ids = {img['id'] for img in test_images}
                
                # 3. Create CVAT Project
                # Always generate project name from dataset (ignore dropdown input)
                project_name = f"{dataset_name}_project"
                
                print(f"🏗️ Creating CVAT project: {project_name}")
                
                # Create labels as dict format for ProjectWriteRequest
                labels = [
                    models.PatchedLabelRequest(
                        name=cat['name'],
                        color="#ff0000",
                        attributes=[]
                    ) 
                    for cat in categories
                ]
                
                project_spec = models.ProjectWriteRequest(
                    name=project_name,
                    labels=labels
                )
                
                if org_id:
                    project_data, _ = client.api_client.projects_api.create(
                        project_write_request=project_spec, 
                        org_id=org_id
                    )
                    project_id = project_data.id
                else:
                    project_data, _ = client.api_client.projects_api.create(
                        project_write_request=project_spec
                    )
                    project_id = project_data.id
                
                print(f"  ✅ Project created (ID: {project_id})")
                
                # 4. Create 3 tasks (Train, Validation, Test)
                if inf_format == 'Classification':
                    subsets = [
                        ("train", train_images, train_ids),
                        ("val", val_images, val_ids),
                        ("test", test_images, test_ids)
                    ]
                else:
                    subsets = [
                        ("Train", train_images, train_ids),
                        ("Validation", val_images, val_ids),
                        ("Test", test_images, test_ids)
                    ]
                
                task_urls = []
                
                for subset_name, subset_images, subset_ids in subsets:
                    print(f"\n📝 Creating task: {dataset_name}_{subset_name}")
                    
                    # Create task (labels inherited from project)
                    task_spec = models.TaskWriteRequest(
                        name=f"{dataset_name}_{subset_name}",
                        project_id=project_id,  # Link to the project you just created
                        subset=subset_name,      # Optional: Group into 'Train', 'Test', or 'Validation'
                        segment_size=0           # Optional: 0 = all frames in one job. 
                    )
                    
                    if org_id:
                        task_data, _ = client.api_client.tasks_api.create(
                            task_write_request=task_spec,
                            org_id=org_id
                        )
                        print(f"  ✓ Task created (ID: {task_data.id})")
                    else:
                        task_data, _ = client.api_client.tasks_api.create(
                            task_write_request=task_spec
                        )
                        print(f"  ✓ Task created (ID: {task_data.id})")

                    high_level_task = client.tasks.retrieve(task_data.id)    
                    
                    # Upload images for this subset
                    print(f"  📤 Uploading {len(subset_images)} images...")
                    image_files = [str(images_dir / img['file_name']) for img in subset_images]
                    high_level_task.upload_data(image_files)
                    
                    # Create temporary COCO file with subset annotations
                    subset_annotations = [ann for ann in annotations if ann['image_id'] in subset_ids]
                    
                    temp_coco = {
                        'images': subset_images,
                        'annotations': subset_annotations,
                        'categories': categories,
                        'info': coco_data.get('info', {})
                    }
                    
                    temp_coco_file = output_dir / f"temp_{subset_name}.json"
                    with open(temp_coco_file, 'w') as f:
                        json.dump(temp_coco, f)
                    
                    # Upload annotations
                    print(f"  📥 Importing {len(subset_annotations)} annotations...")
                    high_level_task.import_annotations(
                        format_name="COCO 1.0",
                        filename=str(temp_coco_file)
                    )
                    
                    # Apply Tag Annotations for Classification
                    if inf_format == "Classification":
                        try:
                            print(f"  🏷️  Applying tag annotations for Classification...")
                            
                            # 1. Retrieve Task Labels to get internal IDs
                            task_labels = high_level_task.get_labels()
                            label_name_to_id = {l.name: l.id for l in task_labels}
                            
                            # 2. Prepare mapping of Image Filename -> Frame Index
                            # Sort by file_name to match CVAT order
                            sorted_subset_images = sorted(subset_images, key=lambda x: x['file_name'])
                            
                            image_id_to_frame = {}
                            for idx, img_info in enumerate(sorted_subset_images):
                                image_id_to_frame[img_info['id']] = idx
                            
                            # 3. Create Tags from Annotations
                            category_map = {c['id']: c['name'] for c in categories}
                            
                            tags_to_create = []
                            
                            for ann in subset_annotations:
                                img_id = ann['image_id']
                                cat_id = ann['category_id']
                                
                                if img_id not in image_id_to_frame:
                                    continue
                                
                                frame_idx = image_id_to_frame[img_id]
                                cat_name = category_map.get(cat_id)
                                
                                # Skip empty category names
                                if not cat_name or cat_name.strip() == '':
                                    continue
                                
                                # Skip combined labels (e.g., "bear cat") for classification
                                if cat_name and ' ' in cat_name:
                                    continue
                                
                                if cat_name and cat_name in label_name_to_id:
                                    cvat_label_id = int(label_name_to_id[cat_name])
                                    
                                    tag_annotation = models.LabeledImageRequest(
                                        frame=int(frame_idx),
                                        label_id=cvat_label_id
                                    )
                                    tags_to_create.append(tag_annotation)
                            
                            if tags_to_create:
                                # 4. Upload Tags to Job 0
                                jobs = high_level_task.get_jobs()
                                if jobs:
                                    target_job = jobs[0]
                                    
                                    from types import SimpleNamespace
                                    patch_request = models.PatchedLabeledDataRequest(
                                        tags=tags_to_create
                                    )
                                    
                                    target_job.update_annotations(patch_request, action=SimpleNamespace(value="create"))
                                    print(f"    ✅ Created {len(tags_to_create)} tag annotations")
                                else:
                                    print("    ⚠️ Warning: No jobs found for task. Skipping tags.")
                            else:
                                print("    ℹ️ No tags to create")
                        except Exception as e:
                            print(f"    ⚠️ Warning: Tag annotation failed: {e}")
                    
                    
                    # Cleanup temp file
                    temp_coco_file.unlink()
                    
                    task_url = f"{url.rstrip('/')}/tasks/{task_data.id}"
                    task_urls.append((subset_name, task_data.id, task_url))
                    print(f"  ✅ Task complete: {task_url}")
                
                # Update task subsets based on subset names
                print("\n🔄 Updating task subsets based on names...")
                
                # Refresh the project to get the newly created tasks
                paginated_data, response_info = client.api_client.tasks_api.list(project_id=int(project_id))
                tasks = paginated_data.results
                
                # Iterate through every task in the project
                for task in tasks:
                    # Get the subset name from the task
                    subset_name = task.subset
                    
                    if not subset_name:
                        print(f"   ⚠️ Skipping task '{task.name}' (No subset assigned)")
                        continue
                    
                    current_subset = None
                    
                    # Check which keyword is in the subset name
                    subset_lower = subset_name.lower()
                    if "train" in subset_lower:
                        current_subset = "Train"
                    elif "val" in subset_lower or "valid" in subset_lower:
                        current_subset = "Validation"
                    elif "test" in subset_lower:
                        current_subset = "Test"
                    
                    # If we found a match and it's different from current, update
                    if current_subset and current_subset != subset_name:
                        print(f"   👉 Found task '{task.name}' with subset '{subset_name}'. Setting to '{current_subset}'")
                        
                        # Use Low-Level API for the patch
                        client.api_client.tasks_api.partial_update(
                            id=task.id,
                            patched_task_write_request=models.PatchedTaskWriteRequest(
                                subset=current_subset
                            )
                        )
                    elif current_subset:
                        print(f"   ✓ Task '{task.name}' already has correct subset: '{current_subset}'")
                    else:
                        print(f"   ⚠️ Skipping task '{task.name}' (No matching subset keyword found)")
                
                print("✅ Task subsets updated successfully.")
                
                
                # Cleanup output directory
                print("\n🧹 Cleaning up temporary files...")
                shutil.rmtree(output_dir.parent, ignore_errors=True)

                self.remove_dataset(dataset_name)

                # Build success message
                project_url = f"{url.rstrip('/')}/projects/{project_id}"
                
                tasks_html = "<br>".join([
                    f"<b>{name}:</b> <a href='{task_url}' target='_blank' style='color: var(--link-text-color); text-decoration: underline;'>Task {task_id}</a>"
                    for name, task_id, task_url in task_urls
                ])
                
                return (
                    f"<div style='padding: var(--size-2); border: 1px solid var(--block-border-color); "
                    f"background: var(--input-background-fill); border-radius: var(--container-radius); "
                    f"color: var(--body-text-color); min-height: 120px;'>"
                    f"<strong>✅ CVAT Project Created Successfully!</strong><br><br>"
                    f"<b>Project:</b> <a href='{project_url}' target='_blank' style='color: var(--link-text-color); text-decoration: underline;'>{project_name} (ID: {project_id})</a><br><br>"
                    f"<b>Tasks Created:</b><br>{tasks_html}"
                    f"</div>"
                )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error creating CVAT project: {str(e)}"
            
            # Cleanup on fail
            if 'temp_extract_dir' in locals(): shutil.rmtree(temp_extract_dir, ignore_errors=True)
            if 'temp_build_dir' in locals(): shutil.rmtree(temp_build_dir, ignore_errors=True)
            
            print(f"Detailed Error: {e}")
            return f"❌ Error during formatting: {str(e)}", None
    
    def upload_to_cvat(self, zip_file_path, dataset_format, progress=None):
        """
        Upload a dataset zipfile that had already been formatted following a standard format to CVAT 
        by creating a project and import the dataset to CVAT using CVAT API. 
        
        Args:
            zip_file_path: Path to the uploaded zip file
            dataset_format: Format string from dropdown ("Detection", "Segmentation", "Classification")
            progress: Optional Gradio Progress object for tracking upload progress
            
        Returns:
            tuple: (success: bool, message: str)
        """

        if not zip_file_path or not os.path.exists(zip_file_path):
            return False, "❌ No zip file provided"
        
        cvat_format = dataset_format
        organization="PixeVision"
        
        try:
            # Step 1: Flatten the zip file (0-25%)
            if progress:
                progress(0.0, desc="📦 Extracting and flattening dataset...")
            dataset_name = Path(zip_file_path).stem
            temp_dir = Path(self.datasets_dir) / f"temp_{dataset_name}"
            
            print(f"📦 Extracting and flattening {dataset_name}...")
            self.extract_and_flatten_zip(zip_file_path, str(temp_dir))
            
            # Step 2: Re-zip the flattened structure (25-50%)
            if progress:
                progress(0.25, desc="📁 Creating zip archive...")
            flattened_zip = Path(self.datasets_dir) / f"{dataset_name}_flattened.zip"
            shutil.make_archive(str(flattened_zip.with_suffix('')), 'zip', str(temp_dir))

            # Clean up temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
            # Step 3: Get CVAT credentials (50-60%)
            if progress:
                progress(0.50, desc="🔗 Connecting to CVAT...")
            url = CVAT_HOST
            username = CVAT_USER
            password = CVAT_PASSWORD
            
            if not all([url, username, password]):
                return False, "❌ Missing CVAT credentials in settings.json"
            
            # Sanitize URL
            host = url.split('/projects')[0].split('/tasks')[0].split('/jobs')[0].rstrip('/')
            
            # Step 4: Create CVAT project and import dataset
            print(f"🔗 Connecting to CVAT at {host}...")
            
            with make_client(host, credentials=(username, password)) as client:
                # Create project (60%)
                if progress:
                    progress(0.60, desc="📁 Creating CVAT project...")
                print(f"📁 Creating CVAT project: {dataset_name} (Org: {organization if organization else 'Personal'})...")
                project_spec = models.ProjectWriteRequest(
                    name=dataset_name,
                )
                
                # Hybrid Approach: 
                # 1. Use low-level API to create project (supports 'org' param reliably)
                if organization:
                    print(f"DEBUG: calling projects_api.create with org={organization}")
                    (project_data, response) = client.api_client.projects_api.create(
                        project_spec, 
                        org=organization
                    )
                    print(f"DEBUG: project_data type: {type(project_data)}")
                    print(f"DEBUG: project_data: {project_data}")
                    
                    if hasattr(project_data, 'id'):
                        project_id = project_data.id
                    elif isinstance(project_data, dict) and 'id' in project_data:
                        project_id = project_data['id']
                    else:
                        print("DEBUG: Could not find id in project_data")
                        project_id = None
                        
                    print(f"DEBUG: extracted project_id: {project_id}")
                else:
                    # Fallback to high-level if no org (or use low-level without org)
                    (project_data, response) = client.api_client.projects_api.create(project_spec)
                    project_id = project_data.id
                
                if project_id is None:
                    return False, "❌ Error: CVAT Project ID is None. Check terminal logs for debug info."

                # 2. Retrieve high-level Project object to use import_dataset helper
                project = client.projects.retrieve(int(project_id))
                
                # Import dataset using project-level method (70-100%)
                if progress:
                    progress(0.70, desc="📥 Uploading dataset to CVAT...")
                print(f"📥 Importing dataset with format: {cvat_format}...")
                project.import_dataset(
                    format_name=cvat_format,
                    filename=str(flattened_zip)
                )
                
                # Update task subsets based on task names (85-95%)
                if progress:
                    progress(0.85, desc="🔄 Updating task subsets...")
                print("🔄 Updating task subsets based on names...")
                
                # Refresh the project to get the newly created tasks
                paginated_data, response_info = client.api_client.tasks_api.list(project_id=int(project_id))
                tasks=paginated_data.results
                
                # Iterate through every task in the project
                for task in tasks:
                    # Convert task name to lowercase for easy matching
                    subset_name = task.subset
                    
                    current_subset = None
                    
                    # Check which keyword is inside the task name
                    if "train" in subset_name:
                        current_subset = "Train"
                    elif "val" in subset_name or "valid" in subset_name:
                        current_subset = "Validation"
                    elif "test" in subset_name:
                        current_subset = "Test"
                    
                    # If we found a match, update the task on the server
                    if current_subset:
                        print(f"   👉 Found task '{task.name}'. Setting subset to '{current_subset}'")
                        
                        # Use Low-Level API for the patch (most reliable)
                        if organization:
                            client.api_client.tasks_api.partial_update(
                                id=task.id,
                                patched_task_write_request=models.PatchedTaskWriteRequest(
                                    subset=current_subset
                                )
                            )
                        else:
                            client.api_client.tasks_api.partial_update(
                                id=task.id,
                                patched_task_write_request=models.PatchedTaskWriteRequest(
                                    subset=current_subset
                                )
                            )
                    else:
                        print(f"   ⚠️ Skipping task '{task.name}' (No matching subset keyword found)")
                
                print("✅ Task subsets updated successfully.")
                
                if progress:
                    progress(1.0, desc="✅ Upload complete!")
                print(f"✅ Successfully uploaded to CVAT project: {dataset_name} (ID: {project_id})")
            
            # Clean up flattened zip
            print(f"🧹 Cleaning up flattened zip: {flattened_zip}")
            if flattened_zip.exists():
                os.remove(flattened_zip)
                print(f"✅ Removed flattened zip: {flattened_zip}")
            else:
                print(f"⚠️ Flattened zip not found at: {flattened_zip}")
            
            return True, f"✅ Successfully created CVAT project '{dataset_name}' (ID: {project_id}) and imported dataset using format '{cvat_format}'"
            
            
        except Exception as e:
            print(f"❌ Error uploading to CVAT: {e}")
            import traceback
            traceback.print_exc()
            
            # Cleanup on error
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                if flattened_zip.exists():
                    os.remove(flattened_zip)
            except:
                pass
                
            return False, f"❌ Error uploading to CVAT: {str(e)}"


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



