import gradio as gr
import json

def create_tab(app):
    with gr.Tab("🤖 Pre-Trained Models") as tab:
        with gr.Column():
            with gr.Accordion("➕ Add Custom Model", open=False):
                with gr.Column():
                    model_name_input = gr.Textbox(
                        label="Model Name", 
                        placeholder="e.g modelABC"
                    )
                    model_uploader = gr.File(
                        label="Upload .pt file",
                        file_types=[".pt"],
                        height=207
                    )
                    upload_model_btn = gr.Button("📤 Upload Model", variant="primary", elem_id="btn")
                    upload_model_status = gr.Markdown(
                        label="Upload Status",
                    )
            own_model_dropdown = gr.Dropdown(
                label="Select Model",
                choices=[],
                interactive=True
            )
            own_delete_btn = gr.Button("🗑️ Delete Model", elem_id="del_btn")
            
            own_model_details = gr.Code(
                label="Model Details",
                interactive=False,
                language="json",
                lines=10,
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🖼️ Run Prediction")
                    own_input_img = gr.Image(
                        label="Input Image", 
                        type="pil",
                        height=400,
                        interactive=True
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
                    own_predict_btn = gr.Button("🚀 Predict Image", variant="primary", elem_id="btn")

                
                with gr.Column():
                    gr.Markdown("### 📊 Prediction Result")
                    own_output_img = gr.Image(label="Prediction Result", type="pil", height=400)
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
        "own_output_img": own_output_img,
        "own_result_details": own_result_details
    }

def setup_events(app, components, all_components):
    c = components
    
    # Internal logic
    def refresh_own_model_dropdown():
        """Refresh dropdown with available custom models"""
        models = app.load_own_models()
        choices = [m["name"] for m in models]
        return gr.update(choices=choices, value=None)
    
    def handle_model_upload(model_name, upload_file):
        """Handle custom model upload event"""
        if not upload_file:
            return "❌ Please upload a .pt file", gr.update()
        
        success, message = app.save_own_model(model_name, upload_file)
        
        # Refresh dropdown
        updated_dropdown = refresh_own_model_dropdown()
        
        return message, updated_dropdown
    
    def handle_model_delete(model_name):
        """Handle model deletion event"""
        if not model_name:
            return "❌ Please select a model", gr.update(), ""
        
        success, message = app.delete_own_model(model_name)
        
        # Refresh dropdown
        updated_dropdown = refresh_own_model_dropdown()
        
        return message, updated_dropdown, ""
    
    def display_model_details(model_name):
        """Display selected model details"""
        if not model_name:
            return ""
        
        details = app.get_own_model_details(model_name)
        return json.dumps(details, indent=2)
    
    def predict_own_model(model_name, image, conf, iou):
        """Run prediction with selected custom model"""
        if not image:
            return None, json.dumps({"error": "Please upload an image"}, indent=2)
        
        if not model_name:
            return image, json.dumps({"error": "Please select a model"}, indent=2)
        
        output_img, details_json = app.predict_with_own_model(
            model_name, image, conf, iou
        )
        
        return output_img, details_json

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
    
    c["own_model_dropdown"].change(
        fn=display_model_details,
        inputs=[c["own_model_dropdown"]],
        outputs=[c["own_model_details"]]
    )
    
    c["own_delete_btn"].click(
        fn=handle_model_delete,
        inputs=[c["own_model_dropdown"]],
        outputs=[c["upload_model_status"], c["own_model_dropdown"], c["own_model_details"]]
    )
    
    c["own_predict_btn"].click(
        fn=predict_own_model,
        inputs=[c["own_model_dropdown"], c["own_input_img"], c["own_conf_slider"], c["own_iou_slider"]],
        outputs=[c["own_output_img"], c["own_result_details"]]
    )
