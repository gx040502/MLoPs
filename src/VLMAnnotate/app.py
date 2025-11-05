import json
import os
import shutil
import zipfile
from pathlib import Path

import torch
import matplotlib
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

from .utils import COCODatasetBuilder, GroundingDINODetector

class APP():
    def __init__(self, vlm_model: GroundingDINODetector=None):
        self.json_path = 'settings.json'
        try:
            with open(self.json_path, 'r') as config_file:
                self.config = json.load(config_file)
                self.datasets_dir = self.config.get("datasets_dir", ".gradio")
                self.html = self.create_dataset_html()
                self.selected_dataset = ''
                self.selected_dataset_1st_img_path = "./Gitlab/.asset/VLM experiment.png"

            self.model = vlm_model

        except (FileNotFoundError, json.JSONDecodeError):
            print("Could not load settings.json. Using default configuration.")
            self.config = {}
            self.datasets_dir = ".gradio"

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
            return True, f"Selected dataset: {name}\nPath: {dataset.get('path', 'N/A')}"
        return False, f"Dataset '{name}' not found."
    
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
            os.makedirs(output_dir, exist_ok=True)
            # Use a context manager to handle the zip file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # Extract all the contents to the specified output directory
                zip_ref.extractall(output_dir)
            

            # Update the config with the new dataset
            if "datasets" not in self.config:
                self.config["datasets"] = []
            self.config["datasets"].append({"name": project_name, "path": output_dir})
            self.save_config()

            return True, f"Successfully unzipped the file to '{output_dir}'.\n\nExtracted files:\n" + "\n".join([f"- {file}" for file in os.listdir(output_dir)])
        except Exception as e:
            return False, f"An error occurred while unzipping the file: {e}"
        
    def get_all_datasets(self, name=None):
        # A list of datasets from the configuration, each element is a dict with 'name' and 'path'
        return self.config.get("datasets", [])

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

    def process_image(self, image, text_prompt, confidence_threshold=0.3):
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
        
    def inference_dataset(self, prompt='.',  confidence_threshold=0.3):
        
        sel_dataset = self.selected_dataset
        dataset_dir = Path(self.get_dataset_by_name(sel_dataset)[1].get('path', ''))

        output_dir = Path(self.datasets_dir).parent / '.output' / f"{self.selected_dataset}_coco"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        coco_builder = COCODatasetBuilder(
                        base_dir=output_dir.as_posix(),
                        contributor="Chee Yee",
                        description="Dataset",
                        version="1.0.0",
                        categories=[
                                    {"id": 1, "name": "object", "supercategory": ""},
                                ]
                    )
        
        imgs = [f for f in dataset_dir.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']]

        for img_path in imgs:
            img = Image.open(img_path).convert("RGB")
            width, height = img.size
            img_id = coco_builder.add_image(Path(img_path).name, width, height, verbose=False)

            annotated_image, text, detections = self.process_image(
                image= img,
                text_prompt= prompt,
                confidence_threshold= confidence_threshold
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

                coco_builder.add_annotation(
                    img_id=img_id,
                    category_id=1,
                    xywh=xywh,
                    verbose=False
                )
        coco_builder.save_json(Path(coco_builder.directories['annotations'])/'instances_Train.json')
        shutil.make_archive(coco_builder.directories['base'], 'zip', coco_builder.directories['base'])

        return f"Inference completed on {len(imgs)} images. Results saved to '{output_dir}'."


if __name__ == "__main__":
    # Load model into device: GPU or CPU
    print('Loading the model and processor...')
    model_id = "IDEA-Research/grounding-dino-base"
    model_id = "./.cache/huggingface/hub/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
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



