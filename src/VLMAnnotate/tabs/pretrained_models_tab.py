import gradio as gr
import json
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path
from PIL import Image
import tempfile
from src.ModelManager.ModelManager import ModelManager
from src.ModelManager.DBmanager import DBManager

def create_tab(app):
    with gr.Tab("🤖 Pre-Trained Models") as tab:
        with gr.Column():
            
            with gr.Row():
                # Left Column: Add Custom Model
                with gr.Column(scale=1):
                    gr.Markdown("### ➕ Add Custom Model")
                    model_uploader = gr.File(
                        label="Upload .pt file",
                        file_types=[".pt"],
                        height=207
                    )
                    model_name_input = gr.Textbox(
                        label="Model Name", 
                        placeholder="e.g modelABC"
                    )
                    upload_model_status = gr.Markdown(
                        value="",
                        visible=True
                    )
                
                # Right Column: Model Selection and Actions
                with gr.Column(scale=1):
                    gr.Markdown("### 📂 Select Model")
                    own_model_dropdown = gr.Dropdown(
                        label="Select Model",
                        choices=[],
                        interactive=True
                    )
                    with gr.Row():
                        upload_model_btn = gr.Button("📤 Upload Model", variant="primary", elem_id="btn")
                        own_delete_btn = gr.Button("🗑️ Delete Model", elem_id="del_btn", visible=False)
            
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
        "own_model_details": own_model_details,
        "own_input_img": own_input_img,
        "own_conf_slider": own_conf_slider,
        "own_iou_slider": own_iou_slider,
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
    
    def handle_model_delete(model_id):
        """Handle model deletion event"""
        if not model_id:
            return "❌ Please select a model", gr.update(), ""
        
        success, message = model_manager.delete_model(model_id)
        
        # Refresh dropdown
        updated_dropdown = refresh_own_model_dropdown()
        
        return message, updated_dropdown, ""
    
    def predict_own_model(model_id, input_files, conf, iou):
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
                                 save_path = os.path.join(output_dir, file)
                                 
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
                output_path, details = model_manager.inference_video(video_path, model, conf, iou)
                if not output_path:
                    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), details
                return gr.update(visible=False), gr.update(visible=True, value=output_path), gr.update(visible=False), details
            except Exception as e:
                 print(f"Prediction Error: {e}")
                 return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), f"Error: {str(e)}"

        # 3. Batch Images
        gallery_results = []
        for file_path in input_files:
            try:
                if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    continue
                image = Image.open(file_path)
                output_img, _ = model_manager.inference_image(image=image, model=model, conf=conf, iou=iou)
                gallery_results.append(output_img)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if not gallery_results:
             return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), json.dumps({"error": "No valid images processed"})
             
        return gr.update(value=gallery_results, visible=True), gr.update(visible=False), gr.update(visible=False), json.dumps({"info": f"Processed {len(gallery_results)} images"}, indent=2)

    # --- Event Handlers ---
    c["tab"].select(
        fn=lambda: app.cleanup_preview(), outputs=None
    ).then(
        fn=refresh_own_model_dropdown,
        outputs=[c["own_model_dropdown"]]
    )
    
    c["upload_model_btn"].click(
        fn=handle_model_upload,
        inputs=[c["model_name_input"], c["model_uploader"]],
        outputs=[c["upload_model_status"], c["own_model_dropdown"]]
    )
    
    def on_model_select(model_id):
        html = model_manager.display_model_details(model_id)
        return html, gr.update(visible=True)

    c["own_model_dropdown"].change(
        fn=on_model_select,
        inputs=[c["own_model_dropdown"]],
        outputs=[c["own_model_details"], c["own_delete_btn"]]
    )
    
    c["own_delete_btn"].click(
        fn=handle_model_delete,
        inputs=[c["own_model_dropdown"]],
        outputs=[c["upload_model_status"], c["own_model_dropdown"], c["own_model_details"]]
    )
    
    c["own_predict_btn"].click(
        fn=predict_own_model,
        inputs=[c["own_model_dropdown"], c["own_input_img"], c["own_conf_slider"], c["own_iou_slider"]],
        outputs=[
            c["own_output_gallery"], 
            c["own_output_video"],   
            c["own_output_file"],
            c["own_result_details"]
        ]
    )
