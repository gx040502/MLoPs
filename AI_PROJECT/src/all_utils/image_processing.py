import cv2
import numpy as np
import io
import shutil
import random
import re
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from .json_utils import COCODatasetBuilder

def draw_bounding_boxes(image, detections, threshold=0.3):
    """
    Draw bounding boxes on the image with labels and confidence scores.
    """
    # Create a copy of the image to draw on
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    
    # Try to load a font (fallback to default if not available)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
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

def process_image(model, sam_model, image, text_prompt, confidence_threshold=0.3, inference_format="Detection"):
    """
    Process image with Grounding DINO for object detection.
    """
    try:
        if image is None:
            return None, "Please upload an image.", ""
        
        if not text_prompt or text_prompt.strip() == "":
            return image, "Please enter a text prompt", ""
        
        if model is None:
            return image, "Model not active", ""

        # Split prompt logic
        prompt_parts = re.split(r'[,.]', text_prompt)
        prompt_parts = [p.strip() for p in prompt_parts if p.strip()]
        
        if len(prompt_parts) <= 1:
            result = model.detect_objects(
                image=image,
                text_prompt=text_prompt,
                threshold=confidence_threshold,
            )
            detections = result.get('detections', [])
        else:
            # Separately detect and merge
            print(f"🔀 Splitting prompt into {len(prompt_parts)} parts: {prompt_parts}")
            all_detections = []
            
            for single_prompt in prompt_parts:
                prompt_with_period = single_prompt if single_prompt.endswith('.') else f"{single_prompt}."
                result = model.detect_objects(
                    image=image,
                    text_prompt=prompt_with_period,
                    threshold=confidence_threshold,
                )
                single_detections = result.get('detections', [])
                all_detections.extend(single_detections)
            
            # NMS Logic
            if all_detections:
                 # Local NMS implementation
                def calculate_iou(box1, box2):
                    x1_min, y1_min, x1_max, y1_max = box1
                    x2_min, y2_min, x2_max, y2_max = box2
                    inter_x_min = max(x1_min, x2_min)
                    inter_y_min = max(y1_min, y2_min)
                    inter_x_max = min(x1_max, x2_max)
                    inter_y_max = min(y1_max, y2_max)
                    if inter_x_max < inter_x_min or inter_y_max < inter_y_min: return 0.0
                    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
                    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
                    union_area = box1_area + box2_area - inter_area
                    return inter_area / union_area if union_area > 0 else 0.0

                iou_threshold = 0.5
                sorted_detections = sorted(all_detections, key=lambda x: x['score'], reverse=True)
                final_detections = []
                while sorted_detections:
                    best_det = sorted_detections.pop(0)
                    final_detections.append(best_det)
                    remaining = []
                    for det in sorted_detections:
                        if calculate_iou(best_det['box'], det['box']) <= iou_threshold:
                            remaining.append(det)
                    sorted_detections = remaining
                detections = final_detections
            else:
                detections = []
        
        if len(detections) == 0:
            return image, f"No objects detected with confidence >= {confidence_threshold}", ""
        
        # Draw bounding boxes (initial)
        image_with_boxes = draw_bounding_boxes(image, detections, confidence_threshold)
        
        # Classification Logic
        if inference_format == "Classification" and detections:
            top_detection = max(detections, key=lambda x: x['score'])
            detections = [top_detection]
            image_with_boxes = draw_bounding_boxes(image, detections, confidence_threshold)

        # Segmentation Logic
        if inference_format == "Segmentation" and detections and sam_model:
             try:
                 bboxes = [det['box'] for det in detections]
                 sam_results = sam_model(image, bboxes=bboxes, verbose=False)
                 
                 if sam_results and sam_results[0].masks:
                     res_plotted = sam_results[0].plot() # numpy BGR
                     image_with_boxes = Image.fromarray(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
                     masks_xy = sam_results[0].masks.xy
                     for i, det in enumerate(detections):
                         if i < len(masks_xy):
                             poly = masks_xy[i].flatten().tolist()
                             det['segmentation'] = [poly]
             except Exception as e:
                 print(f"SAM Error: {e}")

        # Prepare text
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

def inference_dataset(model, sam_model, datasets_dir, selected_dataset_name, dataset_path, prompt='.', confidence_threshold=0.3, inference_format="Detection"):
    """
    Run inference on entire dataset.
    """
    if not selected_dataset_name or not dataset_path:
         return (
            f"<div style='padding: var(--size-2); border: 1px solid var(--block-border-color); "
            f"background: var(--input-background-fill); border-radius: var(--container-radius); "
            f"color: var(--body-text-color); min-height: 80px;'>"
            f"❌ Error: Dataset not selected or not found."
            f"</div>"
        )
    
    dataset_dir = Path(dataset_path)
    output_dir = Path(datasets_dir)/ '.output' / f"{selected_dataset_name}_coco" 
    output_dir.mkdir(parents=True, exist_ok=True)
    
    initial_categories = []
    category_map = {}
        
    coco_builder = COCODatasetBuilder(
                    base_dir=output_dir.as_posix(),
                    contributor="Chee Yee",
                    description="Dataset",
                    version="1.0.0",
                    categories=initial_categories
                )
    
    coco_builder.coco_json['info']['inference_format'] = inference_format
    
    imgs = [f for f in dataset_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']]

    if len(imgs) < 2:
        return (
            f"<div style='padding: var(--size-2); border: 1px solid var(--block-border-color); "
            f"background: var(--input-background-fill); border-radius: var(--container-radius); "
            f"color: var(--body-text-color); min-height: 80px;'>"
            f"❌ Error: At least 2 images are required for inference. Found only {len(imgs)} image(s)."
            f"</div>"
        )

    for img_path in imgs:
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        img_id = coco_builder.add_image(Path(img_path).name, width, height, verbose=False)

        annotated_image, text, detections = process_image(
            model=model,
            sam_model=sam_model,
            image=img,
            text_prompt=prompt,
            confidence_threshold=confidence_threshold,
            inference_format=inference_format
        )

        save_dir = Path(coco_builder.directories['annotated_images'])/ Path(img_path).name
        annotated_image.save(save_dir)

        save_dir = Path(coco_builder.directories['train_images'])/ Path(img_path).name
        img.save(save_dir)

        for i, det in enumerate(detections):
            label_name = det.get('label', 'unknown')
            if not label_name or label_name.strip() == '': continue

            if label_name in category_map:
                cat_id = category_map[label_name]
            else:
                existing_ids = category_map.values()
                next_id = max(existing_ids) + 1 if existing_ids else 1
                coco_builder.add_category(next_id, label_name)
                category_map[label_name] = next_id
                cat_id = next_id

            xyxy = det['box']
            xywh = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])]
            
            coco_builder.add_annotation(
                img_id=img_id,
                category_id=cat_id,
                xywh=xywh,
                verbose=False,
                segmentation=det.get('segmentation', [])
            )
            
    coco_builder.save_json(Path(coco_builder.directories['annotations'])/'instances_Train.json')
    shutil.make_archive(coco_builder.directories['base'], 'zip', coco_builder.directories['base'])

    annotated_images_dir = Path(coco_builder.directories['annotated_images'])
    
    return (
        f"<div style='padding: var(--size-2); border: 1px solid var(--block-border-color); "
        f"background: var(--input-background-fill); border-radius: var(--container-radius); "
        f"color: var(--body-text-color); min-height: 80px;'>"
        f"Inference completed on {len(imgs)} images.<br>Results saved to '<i>{output_dir}</i>'."
        f"</div>",
        str(annotated_images_dir)
    )

def preview_augmentation(images, **kwargs):
    """
    Applies OpenCV-based augmentations to a list of PIL images for preview.
    """
    if not images or images[0] is None:
        return None
        
    cv_images = []
    for img in images:
        if img is not None:
            cv_images.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    
    if not cv_images: return None

    img_cv = cv_images[0] # Default primary
    h, w = img_cv.shape[:2]

    try:
        mosaic_prob = kwargs.get('mosaic', 0.0)
        mixup_prob = kwargs.get('mixup', 0.0)
        cutmix_prob = kwargs.get('cutmix', 0.0)
        copy_paste_prob = kwargs.get('copy_paste', 0.0)

        # 1. Mosaic
        if mosaic_prob > 0.5 and len(cv_images) >= 4:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            xc, yc = w // 2, h // 2 
            img_tl = cv2.resize(cv_images[0], (xc, yc))
            canvas[0:yc, 0:xc] = img_tl
            img_tr = cv2.resize(cv_images[1], (w-xc, yc))
            canvas[0:yc, xc:w] = img_tr
            img_bl = cv2.resize(cv_images[2], (xc, h-yc))
            canvas[yc:h, 0:xc] = img_bl
            img_br = cv2.resize(cv_images[3], (w-xc, h-yc))
            canvas[yc:h, xc:w] = img_br
            img_cv = canvas
        
        # 2. Mixup
        elif mixup_prob > 0.5 and len(cv_images) >= 2:
            im1 = img_cv
            im2 = cv2.resize(cv_images[1], (w, h))
            img_cv = cv2.addWeighted(im1, 0.5, im2, 0.5, 0)

        # 3. CutMix
        elif cutmix_prob > 0.5 and len(cv_images) >= 2:
            im1 = img_cv
            im2 = cv2.resize(cv_images[1], (w, h))
            pw, ph = w // 2, h // 2
            x = random.randint(0, w - pw)
            y = random.randint(0, h - ph)
            img_cv[y:y+ph, x:x+pw] = im2[y:y+ph, x:x+pw]

        # 4. Copy-Paste
        elif copy_paste_prob > 0.5 and len(cv_images) >= 2:
                im2 = cv2.resize(cv_images[1], (w, h))
                pw, ph = int(w * 0.2), int(h * 0.2)
                sx = random.randint(0, w - pw)
                sy = random.randint(0, h - ph)
                patch = im2[sy:sy+ph, sx:sx+pw]
                dx = random.randint(0, w - pw)
                dy = random.randint(0, h - ph)
                img_cv[dy:dy+ph, dx:dx+pw] = patch

        # HSV
        if any(k in kwargs for k in ['hsv_h', 'hsv_s', 'hsv_v']):
            h_gain = kwargs.get('hsv_h', 0.015)
            s_gain = kwargs.get('hsv_s', 0.7)
            v_gain = kwargs.get('hsv_v', 0.4)
            img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV).astype(np.float32)
            hue_shift = (h_gain * 179) * 0.5 
            sat_shift = (s_gain * 255) * 0.5
            val_shift = (v_gain * 255) * 0.5
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_shift) % 180
            img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] + sat_shift, 0, 255)
            img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] + val_shift, 0, 255)
            img_cv = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if kwargs.get('fliplr', 0.0) > 0.5:
            img_cv = cv2.flip(img_cv, 1)

        if kwargs.get('flipud', 0.0) > 0.5:
            img_cv = cv2.flip(img_cv, 0)
        
        deg = kwargs.get('degrees', 0.0)
        trans = kwargs.get('translate', 0.1)
        scale_gain = kwargs.get('scale', 0.5)
        shear_deg = kwargs.get('shear', 0.0)
        persp = kwargs.get('perspective', 0.0)

        if deg != 0 or trans != 0 or scale_gain != 0 or shear_deg != 0 or persp != 0:
            C = np.eye(3)
            C[0, 2] = -w / 2
            C[1, 2] = -h / 2
            s = 1.0 + (scale_gain * 0.5) 
            R = np.eye(3)
            a = np.deg2rad(deg)
            R[0, 0] = R[1, 1] = s * np.cos(a)
            R[0, 1] = -s * np.sin(a)
            R[1, 0] = s * np.sin(a)
            S = np.eye(3)
            S[0, 1] = np.tan(np.deg2rad(shear_deg)) 
            T = np.eye(3)
            T[0, 2] = w / 2 + (trans * w * 0.5)
            T[1, 2] = h / 2 + (trans * h * 0.5)
            M = T @ S @ R @ C  
            
            if persp != 0:
                P = np.eye(3)
                P[2, 0] = persp * 0.001
                P[2, 1] = persp * 0.001
                M = P @ M 
                img_cv = cv2.warpPerspective(img_cv, M, (w, h), borderValue=(114, 114, 114))
            else:
                img_cv = cv2.warpAffine(img_cv, M[:2], (w, h), borderValue=(114, 114, 114))

        erase_prob = kwargs.get('erasing', 0.0)
        if erase_prob > 0:
            ex = int(w * 0.2)
            ey = int(h * 0.2)
            ew = int(w * 0.2)
            eh = int(h * 0.2)
            cv2.rectangle(img_cv, (ex, ey), (ex+ew, ey+eh), (128, 128, 128), -1)

        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
        
    except Exception as e:
        print(f"Augmentation preview error: {e}")
        return images[0] if images else None
