import torch
import re
import json
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor

# Safe import for QWEN
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    try:
        from transformers import AutoModelForCausalLM as Qwen3VLForConditionalGeneration
    except:
        Qwen3VLForConditionalGeneration = None

from huggingface_hub import login, HfFolder
import os
from dotenv import load_dotenv
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')


class QwenVLMDetector:
    """
    A class for object detection using QWEN Vision-Language Model.
    
    Unlike Grounding DINO, QWEN can handle long prompts without splitting
    and returns bounding boxes through conversational responses.
    """
    
    def __init__(self, 
                 model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
                 device: Optional[str] = None,
                 huggingface_token: Optional[str] = None,
                 grounding_dino_model=None):
        """
        Initialize the QWEN VLM detector.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run the model on ('cuda' or 'cpu'). Auto-detected if None.
            huggingface_token: HuggingFace token for authentication
            grounding_dino_model: Optional GroundingDINODetector instance for bounding boxes
        """
        self.model_id = model_id
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.grounding_dino = grounding_dino_model
        
        # Login to HuggingFace if token provided
        token = huggingface_token or HUGGINGFACE_TOKEN
        if token:
            self._login_huggingface(token)
    
    def _login_huggingface(self, token: str):
        """Login to HuggingFace with provided token."""
        try:
            login(token=token)
            print("✅ Logged in to HuggingFace")
        except Exception as e:
            print(f"⚠️ HuggingFace login failed: {e}")
    
    def load_model(self):
        """Load the QWEN model and processor."""
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError("QWEN model not available. Please install transformers>=4.37.0")
        
        print(f"Loading the model and processor: {self.model_id}")
        
        try:
            if self.device == "cuda":
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_id, 
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                ).to(self.device)
            else:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            self._print_device_info()
            print("✅ Model loaded successfully on", self.device)
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _print_device_info(self):
        """Print device information."""
        if self.device == "cuda":
            print("✅ CUDA is available!")
            print(f"🖥️ GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"💾 Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"🔋 Memory Reserved : {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        else:
            print("⚠️ Running on CPU (this will be slow)")
    
    def detect_objects(self, 
                      image: Union[str, Path, Image.Image], 
                      text_prompt: str) -> Dict:
        """
        Detect objects using OCR-based approach for ordered character recognition.
        
        Process:
        1. Ask QWEN to read all characters/text in order (left to right)
        2. Use Grounding DINO to detect all character boxes (generic detection)
        3. Sort boxes from left to right by x-coordinate
        4. Assign QWEN's ordered labels to sorted boxes
        
        Perfect for: License plates, text recognition, sequential labels
        
        Args:
            image: Path to image file or PIL Image object
            text_prompt: Hint for what to read (e.g., "license plate", "text", "characters")
            
        Returns:
            Dictionary containing detection results with ordered labels
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a file path or PIL Image object")
        
        print(f"\n🔍 Using QWEN to read characters in order...")
        
        # Step 1: Ask QWEN to read all characters in order (left to right)
        ordered_characters = self._read_characters_with_qwen(image, text_prompt)
        
        if not ordered_characters:
            return {
                'image': image,
                'detections': [],
                'text_prompt': text_prompt,
                'raw_response': 'No characters identified by QWEN'
            }
        
        print(f"✅ QWEN read (left→right): {' '.join(ordered_characters)}")
        
        # Step 2: Use Grounding DINO to detect all character boxes
        if self.grounding_dino is None:
            print("⚠️ No Grounding DINO model available, cannot get bounding boxes")
            return {
                'image': image,
                'detections': [],
                'text_prompt': text_prompt,
                'raw_response': f"Characters read: {' '.join(ordered_characters)} (no boxes - Grounding DINO not available)"
            }
        
        print(f"🎯 Using Grounding DINO to detect character boxes...")
        
        # Create a more specific prompt for individual characters
        # Instead of generic "text", use all alphanumeric characters
        # This helps Grounding DINO detect each character separately
        all_chars = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9"
        generic_prompt = all_chars
        
        result = self.grounding_dino.detect_objects(
            image=image,
            text_prompt=generic_prompt,
            threshold=0.15  # Even lower threshold for small characters
        )
        
        raw_boxes = result.get('detections', [])
        print(f"  Found {len(raw_boxes)} boxes")
        
        if not raw_boxes:
            print("⚠️ No boxes detected by Grounding DINO")
            return {
                'image': image,
                'detections': [],
                'text_prompt': text_prompt,
                'raw_response': f"Characters read: {' '.join(ordered_characters)} (no boxes detected)"
            }
        
        # Step 3: Sort boxes from left to right
        sorted_boxes = self._sort_boxes_left_to_right(raw_boxes)
        print(f"  Sorted {len(sorted_boxes)} boxes left→right")
        
        # Step 4: Assign QWEN's labels to sorted boxes
        final_detections = self._assign_labels_to_boxes(ordered_characters, sorted_boxes)
        
        print(f"📊 Final detections: {len(final_detections)}")
        
        return {
            'image': image,
            'detections': final_detections,
            'text_prompt': text_prompt,
            'raw_response': f"QWEN read: {' '.join(ordered_characters)}\nMatched {len(final_detections)} boxes"
        }
    
    def _read_characters_with_qwen(self, image: Image.Image, hint: str) -> List[str]:
        """
        Ask QWEN to read all characters in the image from left to right.
        
        Args:
            image: PIL Image
            hint: User's hint about what to read (e.g., "license plate")
            
        Returns:
            List of characters in order (left to right)
        """
        # Construct OCR prompt
        if hint and hint.strip():
            qwen_prompt = f"""Read all characters in this {hint} from left to right.
Return ONLY the characters you see, in order, separated by spaces.
Example: If you see 'ABC123', return: A B C 1 2 3"""
        else:
            qwen_prompt = """Read all text/characters in this image from left to right.
Return ONLY the characters you see, in order, separated by spaces.
Example: If you see 'ABC123', return: A B C 1 2 3"""
        
        # Prepare messages
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": qwen_prompt}
            ]
        }]
        
        try:
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text], 
                images=[image], 
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()
            }
            
            # Run inference
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=256,
                    do_sample=False
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            response = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True
            )[0]
            
            print(f"  QWEN response: {response}")
            
            # Parse response to extract ordered characters
            characters = self._parse_ordered_characters(response)
            
            return characters
            
        except Exception as e:
            print(f"    ❌ Error reading characters: {e}")
            return []
    
    def _parse_ordered_characters(self, response: str) -> List[str]:
        """
        Parse QWEN's response to extract ordered list of characters.
        
        Args:
            response: QWEN's text response
            
        Returns:
            List of characters in order
        """
        # Clean response
        response = response.strip()
        
        # Remove common phrases
        response = re.sub(r'(the characters are|i see|the text is|it says)', '', response, flags=re.IGNORECASE)
        response = re.sub(r'["\']', '', response)  # Remove quotes
        
        # Try space-separated first (preferred format)
        if ' ' in response:
            chars = response.split()
            # Filter out non-alphanumeric (keep only actual characters)
            chars = [c.strip() for c in chars if c.strip() and (c.isalnum() or len(c) == 1)]
            return chars
        
        # If no spaces, split into individual characters
        chars = list(response)
        # Filter out spaces and special chars (except alphanumeric)
        chars = [c for c in chars if c.isalnum()]
        
        return chars
    
    def _sort_boxes_left_to_right(self, boxes: List[Dict]) -> List[Dict]:
        """
        Sort bounding boxes from left to right based on x-coordinate.
        
        Args:
            boxes: List of detection dictionaries with 'box' key
            
        Returns:
            Sorted list of boxes (left to right)
        """
        # Sort by x1 coordinate (left edge of box)
        sorted_boxes = sorted(boxes, key=lambda det: det['box'][0])
        return sorted_boxes
    
    def _assign_labels_to_boxes(self, labels: List[str], boxes: List[Dict]) -> List[Dict]:
        """
        Assign ordered labels to sorted boxes.
        
        Args:
            labels: Ordered list of labels from QWEN (left to right)
            boxes: Sorted list of boxes from Grounding DINO (left to right)
            
        Returns:
            List of detections with assigned labels
        """
        detections = []
        
        # Match labels to boxes (1-to-1 mapping)
        num_matches = min(len(labels), len(boxes))
        
        if len(labels) != len(boxes):
            print(f"  ⚠️ Mismatch: {len(labels)} labels vs {len(boxes)} boxes")
            print(f"  Using first {num_matches} matches")
        
        for i in range(num_matches):
            detection = {
                'label': labels[i],
                'score': boxes[i].get('score', 0.9),
                'box': boxes[i]['box']
            }
            detections.append(detection)
        
        return detections
    
    def _parse_detection_response(self, response: str, image_size: Tuple[int, int]) -> List[Dict]:
        """
        Parse QWEN's text response to extract bounding boxes.
        
        Args:
            response: Raw text response from QWEN
            image_size: (width, height) of the image
            
        Returns:
            List of detections with format:
            [{'label': str, 'score': float, 'box': [x1, y1, x2, y2]}, ...]
        """
        detections = []
        width, height = image_size
        
        # Pattern to match: Object: <name>, Box: [x1, y1, x2, y2]
        # Also try to match variations like "Object: <name> at [x1, y1, x2, y2]"
        patterns = [
            r'Object:\s*([^,\n]+),\s*Box:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
            r'Object:\s*([^,\n]+)\s+at\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
            r'([^:]+):\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
            r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*-\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                
                # Extract label and coordinates based on pattern
                if len(groups) == 5:
                    if pattern.endswith(r'([^\n]+)'):  # Last pattern (coords first)
                        x1, y1, x2, y2, label = groups
                    else:  # Other patterns (label first)
                        label, x1, y1, x2, y2 = groups
                    
                    try:
                        # Convert to integers
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Validate coordinates
                        x1 = max(0, min(x1, width))
                        y1 = max(0, min(y1, height))
                        x2 = max(0, min(x2, width))
                        y2 = max(0, min(y2, height))
                        
                        # Ensure x1 < x2 and y1 < y2
                        if x1 >= x2 or y1 >= y2:
                            continue
                        
                        detections.append({
                            'label': label.strip(),
                            'score': 0.9,  # QWEN doesn't provide confidence scores
                            'box': [x1, y1, x2, y2]
                        })
                    except (ValueError, IndexError):
                        continue
        
        # If no detections found with patterns, try JSON format
        if not detections:
            detections = self._parse_json_response(response, image_size)
        
        return detections
    
    def _parse_json_response(self, response: str, image_size: Tuple[int, int]) -> List[Dict]:
        """Try to parse response as JSON format."""
        detections = []
        width, height = image_size
        
        try:
            # Try to find JSON-like structures
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for item in data:
                    if isinstance(item, dict) and 'box' in item and 'label' in item:
                        box = item['box']
                        if len(box) == 4:
                            x1, y1, x2, y2 = box
                            x1 = max(0, min(int(x1), width))
                            y1 = max(0, min(int(y1), height))
                            x2 = max(0, min(int(x2), width))
                            y2 = max(0, min(int(y2), height))
                            
                            if x1 < x2 and y1 < y2:
                                detections.append({
                                    'label': item['label'],
                                    'score': item.get('score', 0.9),
                                    'box': [x1, y1, x2, y2]
                                })
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        
        return detections
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    """Example usage of the QwenVLMDetector class."""
    
    # Initialize detector
    detector = QwenVLMDetector(
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        huggingface_token=HUGGINGFACE_TOKEN
    )
    
    # Load model
    detector.load_model()
    
    # Test detection
    test_image = "test.jpg"  # Replace with actual image path
    prompt = "person, car, traffic light, bicycle"
    
    results = detector.detect_objects(test_image, prompt)
    
    print(f"\n📊 Detection Results:")
    print(f"Found {len(results['detections'])} objects")
    for det in results['detections']:
        print(f"  - {det['label']}: {det['box']}")
    
    print(f"\n📝 Raw Response:")
    print(results['raw_response'])
    
    # Clean up
    del detector
