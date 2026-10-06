import torch
import re
import gc
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 300

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from transformers.image_utils import load_image
#from swarm_models import BaseMultiModalModel

from pathlib import Path
from huggingface_hub import login, HfFolder

import os
from dotenv import load_dotenv
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

class GroundingDINODetector:
    """
    A class for zero-shot object detection using Grounding DINO model.
    
    This class provides a convenient interface for loading the model,
    processing images, and visualizing detection results.
    """
    
    def __init__(self, 
                 model_id: str = "IDEA-Research/grounding-dino-base",
                 device: Optional[str] = None,
                 huggingface_token: Optional[str] = None):
        """
        Initialize the Grounding DINO detector.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run the model on ('cuda' or 'cpu'). Auto-detected if None.
            huggingface_token: HuggingFace token for authentication
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.font = self._load_font()
        
        # Login to HuggingFace if token provided
        if huggingface_token:
            self._login_huggingface(huggingface_token)
        
        # Load model and processor
        self.load_model()
        
        # Supported image extensions
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"}
    
    def _login_huggingface(self, token: str):
        """Login to HuggingFace with provided token."""
        try:
            print("Logging into Hugging Face...")
            login(token=token)
            print("✅ Successfully logged into Hugging Face")
        except Exception as e:
            print(f"❌ Failed to login to HuggingFace: {e}")
    
    def _load_font(self) -> ImageFont.ImageFont:
        """Load a nice font for text rendering, with fallback."""
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size=16)
        except:
            return ImageFont.load_default()
    
    def load_model(self):
        """Load the Grounding DINO model and processor."""
        print(f'Loading the model and processor: {self.model_id}')
        
        try:
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
            self.model = self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            self._print_device_info()
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def _print_device_info(self):
        """Print device information."""
        if torch.cuda.is_available():
            print("✅ CUDA is available!")
            print(f"🖥️ GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"💾 Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"🔋 Memory Reserved : {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        else:
            print("❌ CUDA is NOT available. Running on CPU.")
    
    def detect_objects(self, 
                      image: Union[str, Path, Image.Image], 
                      text_prompt: str,
                      threshold: float = 0.3) -> Dict:
        """
        Detect objects in an image based on text prompt.
        
        Args:
            image: Path to image file or PIL Image object
            text_prompt: Text description of objects to detect
            threshold: Confidence threshold for detections
            
        Returns:
            Dictionary containing detection results
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a file path or PIL Image object")
        
        # Process inputs
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = self.processor.post_process_grounded_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]
        
        # Convert to more convenient format
        detections = []
        for score, label, box in zip(results["scores"], results["text_labels"], results["boxes"]):
            detections.append({
                'label': label,
                'score': score.item(),
                'box': box.detach().cpu().numpy().astype(int)
            })
        
        return {
            'image': image,
            'detections': detections,
            'text_prompt': text_prompt,
            'threshold': threshold
        }
    
    def visualize_detections(self, 
                           detection_results: Dict,
                           box_color: str = "red",
                           text_color: str = "red",
                           box_width: int = 3,
                           verbose: bool = True) -> Image.Image:
        """
        Visualize detection results on the image.
        
        Args:
            detection_results: Results from detect_objects()
            box_color: Color for bounding boxes
            text_color: Color for text labels
            box_width: Width of bounding box lines
            verbose: Whether to print every result
    
            
        Returns:
            PIL Image with visualized detections
        """
        image = detection_results['image'].copy()
        draw = ImageDraw.Draw(image)
        
        for detection in detection_results['detections']:
            label = detection['label']
            score = detection['score']
            box = detection['box']
            
            # Draw rectangle
            draw.rectangle(box.tolist(), outline=box_color, width=box_width)
            
            # Draw label with score
            label_text = f"{label} ({score:.2f})"
            draw.text((box[0], box[1] - 15), label_text, fill=text_color, font=self.font)

            if verbose:
                print(f"Detected: {label_text} at {box}")
        
        return image
    
    def process_single_image(self, 
                           image_path: Union[str, Path],
                           text_prompt: str,
                           threshold: float = 0.3,
                           save_result: bool = True,
                           output_dir: Union[str, Path] = "results",
                           show_plot: bool = False,
                           verbose: bool = True) -> Dict:
        """
        Process a single image and optionally save/display results.
        
        Args:
            image_path: Path to the image file or PIL Image object
            text_prompt: Text description of objects to detect
            threshold: Confidence threshold
            save_result: Whether to save the annotated image
            output_dir: Directory to save results
            show_plot: Whether to display the plot
            verbose: Whether to print every result
            
        Returns:
            Detection results dictionary
        """
        image_path = Path(image_path)
        # print(f"Processing image: {image_path.name}")
        
        # Detect objects
        results = self.detect_objects(image_path, text_prompt, threshold)
        
        # Visualize detections
        annotated_image = self.visualize_detections(results, verbose=verbose)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.imshow(annotated_image)
        plt.axis("off")
        plt.title(f"{image_path.name} - '{text_prompt}'")
        
        # Save result if requested
        if save_result:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            save_path = output_dir / f"{image_path.stem}_annotated.jpg"
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            print(f"💾 Saved annotated image to: {save_path}")
        
        # Show plot if requested
        if show_plot:
            # # Create plot Again
            # plt.figure(figsize=(10, 8))
            # plt.imshow(annotated_image)
            # plt.axis("off")
            # plt.title(f"{image_path.name} - '{text_prompt}'")
            plt.show()
        else:
            plt.close()
        
        # Clean up memory
        self._cleanup_memory()
        
        return results
    
    def process_image_directory(self, 
                              directory_path: Union[str, Path],
                              text_prompt: str,
                              threshold: float = 0.3,
                              output_dir: Union[str, Path] = "results",
                              max_images: Optional[int] = None,
                              verbose: bool = True) -> List[Dict]:
        """
        Process all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            text_prompt: Text description of objects to detect
            threshold: Confidence threshold
            output_dir: Directory to save results
            max_images: Maximum number of images to process (None for all)
            verbose: Whether to print every result
            
        Returns:
            List of detection results for each image
        """
        directory_path = Path(directory_path)
        all_results = []
        
        # Get all image files
        image_files = [
            f for f in directory_path.iterdir() 
            if f.suffix.lower() in self.image_extensions
        ]
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"Found {len(image_files)} images to process")
        
        for i, img_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {img_path.name}")
            
            try:
                results = self.process_single_image(
                    img_path, text_prompt, threshold, 
                    save_result=True, output_dir=output_dir, show_plot=False, verbose=verbose
                )
                all_results.append(results)
                
            except Exception as e:
                print(f"❌ Error processing {img_path.name}: {e}")
                continue
        
        print(f"\n✅ Completed processing {len(all_results)} images")
        return all_results
    
    def _cleanup_memory(self):
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def get_detection_statistics(self, results_list: List[Dict]) -> Dict:
        """
        Get statistics from multiple detection results.
        
        Args:
            results_list: List of detection results
            
        Returns:
            Dictionary with statistics
        """
        total_detections = sum(len(r['detections']) for r in results_list)
        total_images = len(results_list)
        
        # Count detections per label
        label_counts = {}
        confidence_scores = []
        
        for result in results_list:
            for detection in result['detections']:
                label = detection['label']
                score = detection['score']
                
                label_counts[label] = label_counts.get(label, 0) + 1
                confidence_scores.append(score)
        
        stats = {
            'total_images': total_images,
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / total_images if total_images > 0 else 0,
            'label_counts': label_counts,
            'avg_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            'min_confidence': min(confidence_scores) if confidence_scores else 0,
            'max_confidence': max(confidence_scores) if confidence_scores else 0
        }
        
        return stats
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
        self._cleanup_memory()


if __name__ == "__main__":
    """Example usage of the GroundingDINODetector class."""
    
    # Initialize detector
    detector = GroundingDINODetector(
        model_id="IDEA-Research/grounding-dino-base",
        huggingface_token=HUGGINGFACE_TOKEN
    )
    
    # Process a single image
    single_result = detector.process_single_image(
        image_path= "./.output/globe/27_4_72-MW_1.jpg",
        text_prompt="Yellow Wire attached to a ball bond.",
        threshold=0.1,
        save_result=False,
        show_plot=True
    )
    
    # # Process multiple images in a directory
    # directory_results = detector.process_image_directory(
    #     # directory_path='./Gitlab/.asset/Fresnel/Assembly/dev_sample/IMAGES/ADC52440010/HE_20250625081725',
    #     directory_path="./Gitlab/.asset",
    #     text_prompt="PCB. Wires. Pink squares. Chips. ",
    #     threshold=0.1,
    #     output_dir="results_fresnel",
    #     max_images=50  # Process only first 10 images
    # )
    
    # # Get statistics
    # stats = detector.get_detection_statistics(directory_results)
    # print("\n📊 Detection Statistics:")
    # for key, value in stats.items():
    #     print(f"  {key}: {value}")
    
    # Clean up
    del detector