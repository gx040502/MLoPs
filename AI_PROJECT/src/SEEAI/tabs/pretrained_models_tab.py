import gradio as gr
import json
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path
from PIL import Image
import tempfile
from AI_PROJECT.src.ModelManager.ModelManager import ModelManager
from AI_PROJECT.src.ModelManager.DBmanager import DBManager

def create_tab(app):
    with gr.Tab("🤖 Pre-Trained Models") as tab:
        gr.Markdown("### 📂 Your Custom Models")
        with gr.Column():
            
            with gr.Row():
                with gr.Column(scale=1):
                    own_model_dropdown = gr.Dropdown(
                        label="Select Model",
                        choices=[],
                        interactive=True
                    )
                    with gr.Row():
                        own_delete_btn = gr.Button("🗑️ Delete Model", elem_id="del_btn", visible=False)
                        confirm_delete_btn = gr.Button("✅ Are you sure?", variant="stop", visible=False)
                        cancel_delete_btn = gr.Button("❌ Cancel", visible=False)
            
                with gr.Column(scale=1):
                    with gr.Row():
                        model_uploader = gr.File(
                            label="Upload your custom model (.pt)",
                            file_types=[".pt"],
                            height=120
                        )
                    with gr.Row(scale=1):
                        with gr.Column(scale=1):
                            model_name_input = gr.Textbox(
                                label="Model Name", 
                                placeholder="e.g modelABC"
                            )

                        with gr.Column(scale=1):
                            upload_model_btn = gr.Button("📤 Upload Model", variant="primary", elem_id="btn")

                            upload_model_status = gr.Markdown(
                                value="",
                                visible=True
                            )
                    
                # Right Column: Model Selection and Actions
            # Model Details Accordion (spans full width below)
            with gr.Accordion("📊 Model Details", open=False):
                own_model_details = gr.HTML(
                    value="<p>Select a model to view details</p>"
                )

            
            # Spacer for visual separation
            gr.HTML("""
                <div
                    style="margin: 15px 0; 
                    height: 1px; 
                    background: linear-gradient(90deg, transparent, #6366f1, transparent); 
                    box-shadow: 0 0 10px rgba(99, 102, 241, 0.5);">
                </div>
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🖼️ Run Prediction")
                    own_input_img = gr.File(
                        label="Input Images (Batch), Zip (Images only), or Video", 
                        file_types=["image", "video", ".zip"],
                        file_count="multiple",
                        height=400
                    )
                    with gr.Row():
                        own_conf_slider = gr.Slider(
                            minimum=0.01, maximum=1.0, value=0.25, step=0.01,
                            label="Confidence Threshold"
                        )
                        own_iou_slider = gr.Slider(
                            minimum=0.01, maximum=1.0, value=0.45, step=0.01,
                            label="IOU Threshold"
                        )
                    
                    own_output_ext = gr.Dropdown(
                        label="Output Extension",
                        choices=[".jpg", ".png", ".bmp", ".webp", ".mp4", ".mkv", ".webm"],
                        value=".jpg",
                        interactive=True,
                        info="Select image format for Zip/Batch or video format for Video input"
                    )
                    own_predict_btn = gr.Button("🚀 Predict", variant="primary", elem_id="btn")

                
                with gr.Column():
                    gr.Markdown("### 📊 Prediction Result")
                    own_output_gallery = gr.Gallery(
                        label="Prediction Results", 
                        show_label=True, 
                        elem_id="own_output_gallery", 
                        columns=[3], 
                        rows=[2], 
                        height=400, 
                        object_fit="contain"
                    )
                    own_output_video = gr.Video(label="Prediction Result Video", height=400, visible=False)
                    own_output_file = gr.File(label="Download Results (Zip)", visible=False)
                    with gr.Accordion("📋 Detection Details", open=False):
                        own_result_details = gr.Code(label="Detection Details", language="json", elem_id="detection_details_code", lines=10)

    return {
        "tab": tab,
        "model_name_input": model_name_input,
        "model_uploader": model_uploader,
        "upload_model_btn": upload_model_btn,
        "upload_model_status": upload_model_status,
        "own_model_dropdown": own_model_dropdown,
        "own_delete_btn": own_delete_btn,
        "confirm_delete_btn": confirm_delete_btn,
        "cancel_delete_btn": cancel_delete_btn,
        "own_model_details": own_model_details,
        "own_input_img": own_input_img,
        "own_conf_slider": own_conf_slider,
        "own_iou_slider": own_iou_slider,
        "own_output_ext": own_output_ext,
        "own_predict_btn": own_predict_btn,
        "own_output_gallery": own_output_gallery,
        "own_output_video": own_output_video,
        "own_output_file": own_output_file,
        "own_result_details": own_result_details
    }

def setup_events(app, components, all_components):
    c = components
    model_manager = ModelManager('database.db')

    # Internal logic
    def refresh_own_model_dropdown():
        """Refresh dropdown with available custom models"""
        models = model_manager.get_pretrained_models()
        # Create (label, value) tuples: display name, use id as value
        choices = [(m["name"], m["id"]) for m in models]
        return gr.update(choices=choices, value=None)
    
    def handle_model_upload(model_name, upload_file):
        """Handle custom model upload event"""
        if not upload_file:
            return "❌ Please upload a .pt file", gr.update()
        
        success, message = model_manager.create_pretrained_model(model_name, str(upload_file))

        # Refresh dropdown
        updated_dropdown = refresh_own_model_dropdown()
        
        # Return message to status markdown
        return message, updated_dropdown
    
    def request_model_delete():
        """Switch to confirmation view"""
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

    def cancel_model_delete():
        """Cancel delete and return to normal view"""
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    def handle_model_delete(model_id):
        """Handle model deletion event"""
        if not model_id:
            return "❌ No model selected", gr.update(), "", gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        
        success, message = model_manager.delete_model(model_id)
        
        # Refresh dropdown
        updated_dropdown = refresh_own_model_dropdown()
        
        return message, updated_dropdown, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    def predict_own_model(model_id, input_files, conf, iou, out_ext):
        if not model_id or not input_files:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "Please select a model and upload media"
        
        # Handle both single file (dict) and multiple files (list) from Gradio
        if isinstance(input_files, dict):
            # Single file upload returns a dict with 'path' key
            input_files = [input_files.get('path') or input_files]
        elif not isinstance(input_files, list):
            input_files = [input_files]
        
        # Extract paths if items are dicts
        file_paths = []
        for item in input_files:
            if isinstance(item, dict):
                file_paths.append(item.get('path', item))
            else:
                file_paths.append(item)
        input_files = file_paths
        
        model = model_manager.load_model(model_id)
        if model is None:
             return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), json.dumps({"error": f"Failed to load model ID: {model_id}"})
             
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        import zipfile
        import tempfile
        import shutil
        import os
        import cv2

        # 1. Single Zip File
        if len(input_files) == 1 and input_files[0].lower().endswith('.zip'):
             zip_path = input_files[0]
             try:
                 temp_dir = tempfile.mkdtemp()
                 extract_dir = os.path.join(temp_dir, "input")
                 output_dir = os.path.join(temp_dir, "output")
                 os.makedirs(extract_dir, exist_ok=True)
                 os.makedirs(output_dir, exist_ok=True)
                 
                 with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                     zip_ref.extractall(extract_dir)
                     
                 processed_count = 0
                 for root, dirs, files in os.walk(extract_dir):
                     for file in files:
                         if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                             img_path = os.path.join(root, file)
                             try:
                                 image = Image.open(img_path)
                                 res_img, _ = model_manager.inference_image(image, model, conf, iou)
                                 
                                 # Determine save name based on chosen extension
                                 file_path = Path(file)
                                 if out_ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                                      save_name = file_path.stem + out_ext
                                 else:
                                      save_name = file # Keep original if selected video ext for image
                                      
                                 save_path = os.path.join(output_dir, save_name)
                                 
                                 if isinstance(res_img, Image.Image):
                                     res_img.save(save_path)
                                 else:
                                     if hasattr(res_img, 'shape'):
                                         res_img_bgr = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
                                         cv2.imwrite(save_path, res_img_bgr)
                                 processed_count += 1
                             except Exception as e:
                                 print(f"Failed to process {file}: {e}")
                 
                 if processed_count == 0:
                     return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), json.dumps({"error": "No valid images found in zip"})

                 output_zip = os.path.join(temp_dir, f"predictions_{Path(zip_path).name}")
                 shutil.make_archive(output_zip.replace('.zip', ''), 'zip', output_dir)
                 
                 return gr.update(visible=False), gr.update(visible=False), gr.update(value=output_zip, visible=True), json.dumps({"info": f"Processed {processed_count} images from zip"})
                 
             except Exception as e:
                 return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), json.dumps({"error": str(e)})

        # 2. Single Video File
        if len(input_files) == 1 and Path(input_files[0]).suffix.lower() in video_extensions:
            video_path = input_files[0]
            try:
                print(f"Processing video: {video_path}")
                
                # Use passed out_ext if it's a video format, else default to .webm
                video_ext = out_ext if out_ext.lower() in video_extensions or out_ext.lower() == '.webm' else '.webm'
                
                output_path, details = model_manager.inference_video(video_path, model, conf, iou, output_extension=video_ext)
                if not output_path:
                    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), details
                return gr.update(visible=False), gr.update(visible=True, value=output_path), gr.update(visible=False), details
            except Exception as e:
                 print(f"Prediction Error: {e}")
                 return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), f"Error: {str(e)}"

        # 3. Batch Images
        gallery_results = []
        try:
            temp_dir = tempfile.mkdtemp()
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            processed_count = 0
            
            for file_path in input_files:
                try:
                    if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        continue
                    
                    image = Image.open(file_path)
                    output_img, _ = model_manager.inference_image(image=image, model=model, conf=conf, iou=iou)

                    # Save for download
                    # Determine save name based on chosen extension
                    path_obj = Path(file_path)
                    if out_ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                         save_name = path_obj.stem + out_ext
                    else:
                         save_name = path_obj.name # Keep original if selected video ext for image
                         
                    save_path = os.path.join(output_dir, save_name)
                    
                    if isinstance(output_img, Image.Image):
                        output_img.save(save_path)
                    else:
                        if hasattr(output_img, 'shape'):
                            res_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(save_path, res_img_bgr)
                    gallery_results.append(save_path)
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            
            if not gallery_results:
                 return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), json.dumps({"error": "No valid images processed"})
            
            # Create zip for download
            output_zip = os.path.join(temp_dir, f"batch_predictions")
            shutil.make_archive(output_zip, 'zip', output_dir)
            output_zip_path = output_zip + ".zip"

            return gr.update(value=gallery_results, visible=True), gr.update(visible=False), gr.update(visible=True, value=output_zip_path), json.dumps({"info": f"Processed {len(gallery_results)} images"}, indent=2)

        except Exception as e:
             return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), json.dumps({"error": str(e)})

    def update_output_choices(files):
        """Update output extension dropdown based on input file types"""
        if not files:
            # Revert to all choices if no file
            return gr.update(choices=[".jpg", ".png", ".bmp", ".webp", ".mp4", ".mkv", ".webm"], value=".jpg")
            
        # Check file types
        has_video = False
        has_image = False
        
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
        
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in video_exts:
                has_video = True
            elif ext in image_exts:
                has_image = True
            elif ext == '.zip':
                has_image = True
        
        if has_video:
            # If single video, show video options
            if len(files) == 1 and Path(files[0]).suffix.lower() in video_exts:
                return gr.update(choices=[".mp4", ".mkv", ".webm"], value=".mp4")
            return gr.update(choices=[".mp4", ".mkv", ".webm"], value=".mp4")

        if has_image:
            return gr.update(choices=[".jpg", ".png", ".bmp", ".webp"], value=".jpg")
            
        return gr.update(choices=[".jpg", ".png", ".bmp", ".webp", ".mp4", ".mkv", ".webm"], value=".jpg")

    # --- Event Handlers ---
    c["tab"].select(
        fn=lambda: app.cleanup_preview(), outputs=None
    ).then(
        fn=refresh_own_model_dropdown,
        outputs=[c["own_model_dropdown"]]
    )
    
    # New handler for file upload
    c["own_input_img"].change(
        fn=update_output_choices,
        inputs=[c["own_input_img"]],
        outputs=[c["own_output_ext"]]
    )
    
    c["upload_model_btn"].click(
        fn=handle_model_upload,
        inputs=[c["model_name_input"], c["model_uploader"]],
        outputs=[c["upload_model_status"], c["own_model_dropdown"]]
    )
    
    def on_model_select(model_id):
        if not model_id:
            return "<p>Select a model to view details</p>", gr.update(visible=False)
        html = model_manager.display_model_details(model_id)
        return html, gr.update(visible=True)

    c["own_model_dropdown"].change(
        fn=on_model_select,
        inputs=[c["own_model_dropdown"]],
        outputs=[c["own_model_details"], c["own_delete_btn"]]
    )
    
    # New Delete Flow
    c["own_delete_btn"].click(
        fn=request_model_delete,
        inputs=None,
        outputs=[c["own_delete_btn"], c["confirm_delete_btn"], c["cancel_delete_btn"]]
    )

    c["cancel_delete_btn"].click(
        fn=cancel_model_delete,
        inputs=None,
        outputs=[c["own_delete_btn"], c["confirm_delete_btn"], c["cancel_delete_btn"]]
    )

    c["confirm_delete_btn"].click(
        fn=handle_model_delete,
        inputs=[c["own_model_dropdown"]],
        outputs=[c["upload_model_status"], c["own_model_dropdown"], c["own_model_details"], c["own_delete_btn"], c["confirm_delete_btn"], c["cancel_delete_btn"]]
    )
    
    c["own_predict_btn"].click(
        fn=predict_own_model,
        inputs=[c["own_model_dropdown"], c["own_input_img"], c["own_conf_slider"], c["own_iou_slider"], c["own_output_ext"]],
        outputs=[
            c["own_output_gallery"], 
            c["own_output_video"],   
            c["own_output_file"],
            c["own_result_details"]
        ]
    )
