import cv2
import numpy as np
import shutil
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from .yolo_utils import YOLODatasetBuilder


def draw_bounding_boxes_qwen(image, detections):
    """
    Draw bounding boxes on the image with labels.
    
    Args:
        image: PIL Image object
        detections: List of detections from QWEN
        
    Returns:
        PIL Image with bounding boxes drawn
    """
    # Create a copy to draw on
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Try to load a nice font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Color palette
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788'
    ]
    
    for i, det in enumerate(detections):
        label = det['label']
        box = det['box']  # [x1, y1, x2, y2]
        score = det.get('score', 0.9)
        
        # Convert box to list if it's a numpy array
        if hasattr(box, 'tolist'):
            box = box.tolist()
        # Ensure it's a flat list of 4 coordinates
        if isinstance(box, (list, tuple)) and len(box) == 4:
            box = [int(x) for x in box]
        else:
            print(f"⚠️ Invalid box format for {label}: {box}")
            continue
        
        # Get color
        color = colors[i % len(colors)]
        
        # Draw rectangle - PIL expects [(x1,y1), (x2,y2)] or [x1,y1,x2,y2]
        draw.rectangle(box, outline=color, width=3)
        
        # Draw label background
        label_text = f"{label} ({score:.2f})"
        
        # Get text size
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw background rectangle for text
        text_bg = [
            box[0],
            box[1] - text_height - 8,
            box[0] + text_width + 8,
            box[1]
        ]
        draw.rectangle(text_bg, fill=color)
        
        # Draw text
        draw.text(
            (box[0] + 4, box[1] - text_height - 4),
            label_text,
            fill='white',
            font=font
        )
    
    return img_draw


def process_image_qwen(qwen_model, image, text_prompt):
    """
    Process image with QWEN for object detection.
    
    Args:
        qwen_model: QwenVLMDetector instance
        image: PIL Image object
        text_prompt: Text description of objects to detect
        
    Returns:
        Tuple of (annotated_image, detection_info, detections)
    """
    try:
        if image is None:
            return None, "Please upload an image.", []
        
        if not text_prompt or text_prompt.strip() == "":
            return image, "Please enter a text prompt", []
        
        if qwen_model is None:
            return image, "QWEN model not loaded", []
        
        print(f"\n🔍 process_image_qwen called with prompt: '{text_prompt}'")
        
        # Run QWEN detection
        result = qwen_model.detect_objects(
            image=image,
            text_prompt=text_prompt
        )
        
        print(f"📊 Result keys: {result.keys()}")
        detections = result.get('detections', [])
        print(f"📊 Number of detections: {len(detections)}")
        
        if len(detections) == 0:
            raw_response = result.get('raw_response', 'No response')
            print(f"⚠️ No detections! Raw response: {raw_response}")
            return image, f"No objects detected. QWEN response: {raw_response}", []
        
        print(f"✅ Drawing boxes for {len(detections)} detections")
        for i, det in enumerate(detections):
            print(f"  Detection {i+1}: {det['label']} at {det['box']}")
        
        # Draw bounding boxes
        image_with_boxes = draw_bounding_boxes_qwen(image, detections)
        
        # Prepare detection info text
        detection_info = []
        for i, det in enumerate(detections):
            label = det['label']
            score = det.get('score', 0.9)
            box = det['box']
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            detection_info.append(
                f"Detection {i+1}:\n"
                f"  Label: {label}\n"
                f"  Confidence: {score:.2f}\n"
                f"  Box: [{x1}, {y1}, {x2}, {y2}]\n"
                f"  Size: {width}x{height} pixels\n"
            )
        
        info_text = "\n".join(detection_info)
        info_text += f"\n\nTotal detections: {len(detections)}"
        
        return image_with_boxes, info_text, detections
        
    except Exception as e:
        print(f"❌ Exception in process_image_qwen: {e}")
        import traceback
        traceback.print_exc()
        return image, f"Error during detection: {str(e)}", []


def inference_dataset_qwen(qwen_model, datasets_dir, selected_dataset_name, dataset_path, prompt='.'):
    """
    Run QWEN inference on entire dataset and prepare for YOLO export.
    
    Args:
        qwen_model: QwenVLMDetector instance
        datasets_dir: Base directory for datasets
        selected_dataset_name: Name of the selected dataset
        dataset_path: Path to the dataset directory
        prompt: Text prompt for detection
        
    Returns:
        Tuple of (status_html, annotated_dir_path)
    """
    if not selected_dataset_name or not dataset_path:
        return (
            f"<div style='padding: var(--size-2); border: 1px solid var(--block-border-color); "
            f"background: var(--input-background-fill); border-radius: var(--container-radius); "
            f"color: var(--body-text-color); min-height: 80px;'>"
            f"❌ Error: Dataset not selected or not found."
            f"</div>",
            ""
        )
    
    dataset_dir = Path(dataset_path)
    output_dir = Path(datasets_dir) / '.output' / f"{selected_dataset_name}_qwen"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create annotated images directory
    annotated_dir = output_dir / 'annotated_images'
    annotated_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all images
    imgs = [f for f in dataset_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']]
    
    if len(imgs) < 1:
        return (
            f"<div style='padding: var(--size-2); border: 1px solid var(--block-border-color); "
            f"background: var(--input-background-fill); border-radius: var(--container-radius); "
            f"color: var(--body-text-color); min-height: 80px;'>"
            f"❌ Error: No images found in dataset."
            f"</div>",
            ""
        )
    
    # Storage for all detections
    all_detections = {}  # image_id -> list of detections
    all_images = {}  # image_id -> image_info
    class_names_set = set()
    
    print(f"\n🔍 Running QWEN inference on {len(imgs)} images...")
    
    for idx, img_path in enumerate(imgs):
        print(f"  [{idx+1}/{len(imgs)}] Processing {img_path.name}...")
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        
        # Run QWEN detection
        annotated_image, text, detections = process_image_qwen(
            qwen_model=qwen_model,
            image=img,
            text_prompt=prompt
        )
        
        # Save annotated image
        save_path = annotated_dir / img_path.name
        annotated_image.save(save_path)
        
        # Store image info
        all_images[idx] = {
            'id': idx,
            'file_name': img_path.name,
            'path': str(img_path),
            'width': width,
            'height': height
        }
        
        # Store detections
        all_detections[idx] = detections
        
        # Collect class names
        for det in detections:
            class_names_set.add(det['label'])
    
    # Convert class names to sorted list
    class_names = sorted(list(class_names_set))
    
    if not class_names:
        return (
            f"<div style='padding: var(--size-2); border: 1px solid var(--block-border-color); "
            f"background: var(--input-background-fill); border-radius: var(--container-radius); "
            f"color: var(--body-text-color); min-height: 80px;'>"
            f"❌ Error: No objects detected in any image. Try a different prompt."
            f"</div>",
            ""
        )
    
    # Save metadata for YOLO export
    metadata = {
        'all_images': all_images,
        'all_detections': all_detections,
        'class_names': class_names,
        'dataset_name': selected_dataset_name,
        'output_dir': str(output_dir)
    }
    
    # Save metadata as JSON
    import json
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    total_detections = sum(len(dets) for dets in all_detections.values())
    
    return (
        f"<div style='padding: var(--size-2); border: 1px solid var(--block-border-color); "
        f"background: var(--input-background-fill); border-radius: var(--container-radius); "
        f"color: var(--body-text-color); min-height: 80px;'>"
        f"✅ Inference completed on {len(imgs)} images.<br>"
        f"Found {total_detections} objects across {len(class_names)} classes.<br>"
        f"Classes: {', '.join(class_names)}<br>"
        f"Results saved to '<i>{output_dir}</i>'.<br>"
        f"<b>Ready to export YOLO format!</b>"
        f"</div>",
        str(annotated_dir)
    )


def export_yolo_format(datasets_dir, selected_dataset_name):
    """
    Export the inference results to YOLO format.
    
    Args:
        datasets_dir: Base directory for datasets
        selected_dataset_name: Name of the selected dataset
        
    Returns:
        Status message HTML
    """
    # Load metadata
    output_dir = Path(datasets_dir) / '.output' / f"{selected_dataset_name}_qwen"
    metadata_path = output_dir / 'metadata.json'
    
    if not metadata_path.exists():
        return (
            f"<div style='padding: var(--size-2); border: 1px solid var(--block-border-color); "
            f"background: var(--input-background-fill); border-radius: var(--container-radius); "
            f"color: var(--body-text-color); min-height: 80px;'>"
            f"❌ Error: No inference results found. Please run 'Inference Full Dataset' first."
            f"</div>"
        )
    
    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    all_images = metadata['all_images']
    all_detections = metadata['all_detections']
    class_names = metadata['class_names']
    
    # Create YOLO dataset
    yolo_dir = Path(datasets_dir) / '.output' / f"{selected_dataset_name}_yolo"
    
    # Remove if exists
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    
    print(f"\n📦 Creating YOLO dataset at: {yolo_dir}")
    
    builder = YOLODatasetBuilder(
        base_dir=yolo_dir,
        class_names=class_names,
        train_val_split=0.8
    )
    
    # Add all images and annotations
    for img_id_str, img_info in all_images.items():
        img_id = int(img_id_str)
        img_path = Path(img_info['path'])
        
        # Add image
        builder.add_image(img_path, image_id=img_id)
        
        # Add annotations
        detections = all_detections.get(str(img_id), [])
        for det in detections:
            builder.add_annotation(
                image_id=img_id,
                class_name=det['label'],
                bbox_xyxy=det['box']
            )
    
    # Split and save
    builder.split_and_save()
    
    # Print statistics
    stats = builder.get_statistics()
    
    # Create ZIP archive
    shutil.make_archive(str(yolo_dir), 'zip', yolo_dir)
    
    return (
        f"<div style='padding: var(--size-2); border: 1px solid var(--block-border-color); "
        f"background: var(--input-background-fill); border-radius: var(--container-radius); "
        f"color: var(--body-text-color); min-height: 80px;'>"
        f"✅ YOLO Dataset Created Successfully!<br><br>"
        f"<b>Statistics:</b><br>"
        f"• Total Images: {stats['total_images']}<br>"
        f"• Total Annotations: {stats['total_annotations']}<br>"
        f"• Classes: {', '.join(class_names)}<br><br>"
        f"<b>Output Location:</b><br>"
        f"<i>{yolo_dir}</i><br><br>"
        f"<b>Files Created:</b><br>"
        f"• images/train/ and images/val/<br>"
        f"• labels/train/ and labels/val/<br>"
        f"• data.yaml<br>"
        f"• classes.txt<br>"
        f"• {selected_dataset_name}_yolo.zip<br>"
        f"</div>"
    )
