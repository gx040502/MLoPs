import torch
import re
import gc
# import cv2
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import torch
import os
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
matplotlib.rcParams['figure.dpi'] = 300

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from transformers.image_utils import load_image
from swarm_models import BaseMultiModalModel

# import sys
# sys.path.append(Path('/home/cy/Gitlab').as_posix())
# from huggingface_hub import login, HfFolder
# from config import HUGGINGFACE_TOKEN
# print("Logging into Hugging Face...")
# HfFolder.save_token(HUGGINGFACE_TOKEN)
# login(token=HUGGINGFACE_TOKEN)
# print("Logged into Hugging Face")


class GdinoModel():
    def __init__(self, model_id = "IDEA-Research/grounding-dino-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)

    def draw_bounding_boxes(self, image, detections, labels, threshold=0.3):
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
        for i, (box, score, label) in enumerate(zip(detections["boxes"], detections["scores"], labels)):
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
            
            # Convert image to RGB
            image = image.convert("RGB")
            
            # Process inputs
            inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
            
            # Move inputs to same device as model
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=confidence_threshold,
                text_threshold=0.25,
                target_sizes=[image.size[::-1]]  # (height, width)
            )[0]
            
            # Extract results
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = results["text_labels"]
            
            # Filter by confidence threshold
            valid_indices = scores >= confidence_threshold
            boxes = boxes[valid_indices]
            scores = scores[valid_indices]
            labels = [labels[i] for i in range(len(labels)) if valid_indices[i]]
            
            if len(boxes) == 0:
                return image, f"No objects detected with confidence >= {confidence_threshold}", ""
            
            # Prepare detection results for display
            detection_info = []
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                x1, y1, x2, y2 = box
                detection_info.append(f"Object {i+1}: {label} (confidence: {score:.3f})")
                detection_info.append(f"  Bounding box: ({x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f})")
            
            detection_text = "\n".join(detection_info)
            
            # Draw bounding boxes
            image_with_boxes = self.draw_bounding_boxes(
                image, 
                {"boxes": boxes, "scores": scores}, 
                labels, 
                confidence_threshold
            )
            
            # Prepare raw results
            raw_results = f"Detected {len(boxes)} objects:\n{detection_text}"
            
            return image_with_boxes, detection_text, raw_results
            
        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            print(error_msg)
            return image if image else None, error_msg, ""

# Launch the app
if __name__ == "__main__":

    # Load model into device: GPU or CPU
    print('Loading the model and processor...')
    model_id = "IDEA-Research/grounding-dino-base"
    model_id = "/home/cy/.cache/huggingface/hub/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
    gdino = GdinoModel(model_id)

    img_path = '/home/cy/Gitlab/.asset/HE_9_7_O.jpg'
    image = Image.open(img_path).convert("RGB")
    text_prompt = "Car."
    confidence_threshold = 0.3
    process_result = gdino.process_image(image, text_prompt, confidence_threshold)
    print(process_result)