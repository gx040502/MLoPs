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
                        label="Input Image or Video", 
                        file_types=["image", "video"],
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
                    own_output_preview = gr.Image(label="Prediction Preview", type="pil", height=400, visible=False)
                    own_output_video = gr.Video(label="Prediction Result Video", height=400, visible=False)
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
        "own_output_preview": own_output_preview,
        "own_output_video": own_output_video,
        "own_result_details": own_result_details
    }

def setup_events(app, components, all_components):
    c = components
    model_manager = ModelManager('database.db')

    # Internal logic
    def refresh_own_model_dropdown():
        """Refresh dropdown with available custom models"""
        models = model_manager.get_pretrained_models()
        choices = [m["id"] for m in models]
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
    
    def predict_own_model(model_id, input_file, conf, iou):
        if not model_id or not input_file:
            # Hide both outputs on error
            return gr.update(visible=False), gr.update(visible=False), "Please select a model and upload media"
        
        model = model_manager.load_model(model_id)
        if model is None:
             return gr.update(visible=False), gr.update(visible=False), json.dumps({"error": f"Failed to load model from path: {model_info.get('storage_path')}"})
        file_ext = Path(input_file).suffix.lower()
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        try:
            if file_ext in video_extensions:
                # Process video
                print(f"Processing video: {file_ext}")
                output_path, details = model_manager.inference_video(
                    input_file, model, conf, iou
                )
                
                # Show File, Hide Image
                return gr.update(visible=False), gr.update(visible=True, value=output_path), details
            else:
                # Process image
                print(f"Processing image: {file_ext}")
                image = Image.open(input_file)
    
                output_img, details_json = model_manager.inference_image(
                    image=image,model=model, conf=conf, iou=iou
                )
                
                # Show Image, Hide File (pass PIL image directly)
                return gr.update(visible=True, value=output_img), gr.update(visible=False), details_json
        
        except Exception as e:
             # Hide both on error
             print(f"Prediction Error: {e}")
             return gr.update(visible=False), gr.update(visible=False), f"Error: {str(e)}"

    # --- Event Handlers ---
    c["tab"].select(
        fn=lambda _: app.cleanup_preview(), outputs=None
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
            c["own_output_preview"], # Image component
            c["own_output_video"],   # Video component
            c["own_result_details"]
        ]
    )
