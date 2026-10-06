import os
import shutil
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO, SAM

from src.seeai.data.db_manager import DBManager
from src.seeai.models.grounding_dino import GroundingDINODetector
from src.seeai.utils.file_utils import extract_and_flatten_zip, upload_dataset_by_zip
from src.seeai.config.settings import CVAT_HOST_IP, CVAT_HOST_PORT, CVAT_USER, CVAT_PASSWORD

# Utility Modules
from src.seeai.data import dataset_manager as dataset_utils
from src.seeai.utils import video_processing as video_utils
from src.seeai.utils import training_utils
from src.seeai.utils import image_processing
from src.seeai.integrations import cvat_client as cvat_utils
from src.seeai.utils import qwen_processing

CVAT_HOST = str(CVAT_HOST_IP) + ":" + str(CVAT_HOST_PORT)

class APP():

    def __init__(self, vlm_model: GroundingDINODetector=None, qwen_model=None):
        # In-memory dataset storage (dict format: {name: {name, path}})
        self.datasets = {}
        # Go up 3 levels from src/seeai/core/annotation_engine.py to the pipeline root, then into data/datasets/temp
        self.datasets_dir = str(Path(__file__).resolve().parents[3] / "data/datasets/temp")
        self.selected_dataset = ''
        self.selected_dataset_1st_img_path = ""

        # Models
        self.sam_model = SAM("data/weights/sam2.1_b.pt")
        self.model = vlm_model
        self.qwen_model = qwen_model
    
    # -------------------------------------------------------------------------
    #                         DATASET MANAGEMENT
    # -------------------------------------------------------------------------
    def get_all_datasets(self, name=None):
        return list(self.datasets.values())

    def get_dataset_by_name(self, name):
        dataset = self.datasets.get(name)
        if dataset:
            return True, dataset
        return False, f"Dataset '{name}' not found."    
    
    def remove_dataset(self, name):
        success, dataset_to_remove = self.get_dataset_by_name(name)
        if not success:
            return False, f"Dataset '{name}' not found."

        dataset_path = dataset_to_remove.get("path")
        
        # Call utility to remove files
        success, msg = dataset_utils.remove_dataset_files(dataset_path)
        
        if success or "Directory" in msg: # Even if zip fail, remove from memory
             del self.datasets[name]
             if self.selected_dataset == name:
                 self.selected_dataset = ''
                 self.selected_dataset_1st_img_path = ''

        return success, msg

    def select_dataset(self, name):
        success, dataset = self.get_dataset_by_name(name)
        if success:
            self.selected_dataset = name
            dataset_path = dataset.get("path", "")
            
            # Find first image for preview
            self.selected_dataset_1st_img_path = ''
            if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
                for file in os.listdir(dataset_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        self.selected_dataset_1st_img_path = os.path.join(dataset_path, file)
                        break
            
            msg = f"Path: {dataset.get('path', 'N/A')}"
            return True, msg
        return False, f"Dataset '{name}' not found."
    
    # -------------------------------------------------------------------------
    #                         ZIPFILE LOGIC
    # -------------------------------------------------------------------------
    def extract_and_flatten_zip(self, zip_path, extract_to):
        return extract_and_flatten_zip(zip_path, extract_to)

    def upload_dataset_by_zip(self, zip_file_path):
        success, message, dataset_info = upload_dataset_by_zip(
            zip_file_path,
            self.datasets_dir
        )
        if success and dataset_info:
            self.datasets[dataset_info["name"]] = dataset_info
        
        return success, message, dataset_info.get("name", "") if dataset_info else ""

    def inspect_dataset_zip(self, zip_path):
        return dataset_utils.inspect_dataset_zip(zip_path, self.datasets_dir)

    # -------------------------------------------------------------------------
    #                         VIDEO LOGIC
    # -------------------------------------------------------------------------
    def scan_for_videos(self, dataset_name):
        success, dataset = self.get_dataset_by_name(dataset_name)
        if not success: return []
        return video_utils.scan_for_videos(dataset.get("path"))

    def extract_frames_from_dataset(self, dataset_name, video_config_df, interval_val=1.0):
        success, dataset = self.get_dataset_by_name(dataset_name)
        if not success: return False, "Dataset not found"

        success_extract, result = video_utils.extract_frames_from_dataset(
            dataset_path=dataset.get("path"),
            dataset_name=dataset_name,
            datasets_dir=self.datasets_dir,
            video_config_df=video_config_df,
            interval_val=interval_val
        )

        if success_extract:
             # zip_output_path is result
             # Auto-upload extracted dataset
             success_up, msg, info = self.upload_dataset_by_zip(result)
             if success_up:
                 # Clean up zip
                 try: os.remove(result)
                 except: pass
                 # Auto select new dataset
                 new_name = info
                 self.select_dataset(new_name)
                 return True, new_name
             else:
                 return False, f"Extraction worked but upload failed: {msg}"
        else:
            return False, result

    # -------------------------------------------------------------------------
    #                         MODEL / TRAINING LOGIC
    # -------------------------------------------------------------------------
    def get_model_plots(self, model_path):
        return training_utils.get_model_plots(model_path)

    def get_format_pretrained_models(self, format_name=None):
        return training_utils.get_format_pretrained_models(format_name)

    def get_random_sample_images(self, project_id, count=4):
        return training_utils.get_random_sample_images(project_id, self.datasets_dir, count)
    
    def preview_augmentation(self, images, **kwargs):
        return image_processing.preview_augmentation(images, **kwargs)

    # -------------------------------------------------------------------------
    #                         INFERENCE LOGIC
    # -------------------------------------------------------------------------
    def draw_bounding_boxes(self, image, detections, threshold=0.3):
        return image_processing.draw_bounding_boxes(image, detections, threshold)

    def process_image(self, image, text_prompt, confidence_threshold=0.3, inference_format="Detection"):
        return image_processing.process_image(
            model=self.model,
            sam_model=self.sam_model,
            image=image,
            text_prompt=text_prompt,
            confidence_threshold=confidence_threshold,
            inference_format=inference_format
        )
            
    def inference_dataset(self, prompt='.', confidence_threshold=0.3, inference_format="Detection"):
        success, dataset = self.get_dataset_by_name(self.selected_dataset)
        if not success:
             return f"❌ Error: Dataset '{self.selected_dataset}' not found.", ""
        
        return image_processing.inference_dataset(
            model=self.model,
            sam_model=self.sam_model,
            datasets_dir=self.datasets_dir,
            selected_dataset_name=self.selected_dataset,
            dataset_path=dataset.get('path'),
            prompt=prompt,
            confidence_threshold=confidence_threshold,
            inference_format=inference_format
        )

    # -------------------------------------------------------------------------
    #                         CVAT LOGIC
    # -------------------------------------------------------------------------
    
    def create_cvat_project_with_tasks(self):
        result = cvat_utils.create_cvat_project_with_tasks(
            datasets_dir=self.datasets_dir,
            selected_dataset_name=self.selected_dataset
        )
        
        # If success (or partial), we might want to cleanup memory if file was deleted
        # The util deletes the file. We should check if path exists.
        # But simple way: just try to remove strictly from memory if we know it deleted it.
        # But parsing result string is hard. 
        # Check if local directory exists, if not remove from memory.
        if self.selected_dataset:
             success, ds = self.get_dataset_by_name(self.selected_dataset)
             if success and not os.path.exists(ds.get('path', '')):
                 del self.datasets[self.selected_dataset]
                 self.selected_dataset = ''
                 
        return result
    
    def upload_to_cvat(self, zip_file_path, dataset_format, progress=None):
        return cvat_utils.upload_to_cvat(
            datasets_dir=self.datasets_dir,
            zip_file_path=zip_file_path,
            dataset_format=dataset_format,
            progress=progress
        )

    # -------------------------------------------------------------------------
    #                         QWEN LOGIC
    # -------------------------------------------------------------------------
    
    def process_image_qwen(self, image, text_prompt):
        """Process image with QWEN VLM for object detection."""
        return qwen_processing.process_image_qwen(
            qwen_model=self.qwen_model,
            image=image,
            text_prompt=text_prompt
        )
    
    def inference_dataset_qwen(self, prompt='.'):
        """Run QWEN inference on entire dataset."""
        success, dataset = self.get_dataset_by_name(self.selected_dataset)
        if not success:
            return f"❌ Error: Dataset '{self.selected_dataset}' not found.", ""
        
        return qwen_processing.inference_dataset_qwen(
            qwen_model=self.qwen_model,
            datasets_dir=self.datasets_dir,
            selected_dataset_name=self.selected_dataset,
            dataset_path=dataset.get('path'),
            prompt=prompt
        )
    
    def export_yolo_format(self):
        """Export inference results to YOLO format."""
        return qwen_processing.export_yolo_format(
            datasets_dir=self.datasets_dir,
            selected_dataset_name=self.selected_dataset
        )

    # Cleanup preview
    def cleanup_preview(self):
        """Clean up any temporary preview files directly"""
        pass # Placeholder if needed
