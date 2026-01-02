import gradio as gr
import json
import os
from ultralytics import YOLO
from modelDatabase import ModelRegistry
from datetime import datetime

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
            with gr.Row():
                with gr.Column(scale=1):
                    own_model_dropdown = gr.Dropdown(
                        label="Select Model",
                        choices=[],
                        interactive=True
                    )
                    own_delete_btn = gr.Button("🗑️ Delete Model", elem_id="del_btn")
                with gr.Column(scale=2):
                    with gr.Accordion("📊 Model Details", open=False):
                        own_model_details = gr.HTML(
                            value="<p>Select a model to view details</p>"
                        )
            
            # Spacer for visual separation
            gr.HTML("<div style='margin: 20px 0;'></div>")
            
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
        """Display selected model details with rich HTML formatting"""
        if not model_name:
            return "<p>Select a model to view details</p>"
        
        try:
            # Get model path from config
            own_models = app.load_own_models()
            model_entry = next((m for m in own_models if m.get("name") == model_name), None)
            
            if not model_entry:
                return "<p>Model not found in configuration</p>"
            
            model_path = model_entry.get('path', '')
            
            if not model_path or not os.path.exists(model_path):
                return "<p>Model file not found</p>"
                
            # Load model and get info
            model = YOLO(model_path)
            registry = ModelRegistry()
            model_info = registry.get_pt_model_info(model)
            
            # Format labels - each on new line
            labels = model_info['labels']
            label_str = '\n'.join([f"{k}: {v}" for k, v in labels.items()]) if isinstance(labels, dict) else str(labels)
            
            # Calculate score percentage
            score_pct = model_info.get('primary_score', 0) * 100
            score_color = "#34d399" if score_pct > 80 else "#fbbf24" if score_pct > 50 else "#f87171"
            
            # Format trained date to human-readable
            trained_at_raw = model_info.get('trained_at', 'N/A')
            if trained_at_raw != 'N/A':
                try:
                    # Parse ISO format datetime
                    dt = datetime.fromisoformat(trained_at_raw.replace('Z', '+00:00'))
                    trained_at_formatted = dt.strftime('%b %d, %Y %I:%M %p')
                except:
                    trained_at_formatted = trained_at_raw
            else:
                trained_at_formatted = 'N/A'
            
            # Format metrics
            metrics = model_info.get('metrics', {})
            
            html = f"""
            <style>
                .model-dashboard {{
                    font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
                    color: #e5e7eb;
                    max-width: 100%;
                }}
                .header-section {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 20px;
                    gap: 12px;
                }}
                .model-title {{ 
                    font-size: 1.5rem; 
                    font-weight: 700; 
                    margin: 0; 
                    color: #ffffff;
                }}
                .info-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: 16px;
                }}
                .info-card {{
                    background: #1f2937;
                    border: 1px solid #374151;
                    border-radius: 12px;
                    padding: 16px;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
                }}
                .card-title {{
                    font-size: 0.95rem; 
                    font-weight: 600; 
                    color: #9ca3af;
                    margin-bottom: 12px; 
                    text-transform: uppercase; 
                    letter-spacing: 0.05em;
                    border-bottom: 1px solid #374151;
                    padding-bottom: 8px;
                }}
                .data-row {{
                    display: flex; 
                    justify-content: space-between;
                    margin-bottom: 8px; 
                    font-size: 0.9rem;
                }}
                .data-label {{ color: #d1d5db; }}
                .data-value {{ font-weight: 500; color: #ffffff; text-align: right; }}
                .progress-bg {{
                    background: #374151;
                    height: 8px; 
                    width: 100%; 
                    border-radius: 4px; 
                    margin-top: 6px; 
                    overflow: hidden;
                }}
                .progress-fill {{ 
                    height: 100%; 
                    border-radius: 4px; 
                    transition: width 0.3s ease; 
                    box-shadow: 0 0 8px {score_color};
                }}
                .classes-box {{
                    background: #111827;
                    padding: 8px;
                    border: 1px solid #374151;
                    border-radius: 6px; 
                    font-size: 0.8rem; 
                    color: #9ca3af;
                    white-space: pre-line; 
                    max-height: 150px;
                    overflow-y: auto;
                }}
            </style>
            
            <div class="model-dashboard">
                <div class="header-section">
                    <h3 class="model-title">{model_name}</h3>
                </div>
                
                <div class="info-grid">
                    <div class="info-card">
                        <div class="card-title">🎯 Performance</div>
                        <div class="data-row">
                            <span class="data-label">Task</span>
                            <span class="data-value">{model_info['task'].title()}</span>
                        </div>
                        <div style="margin-bottom: 12px;">
                            <div class="data-row" style="margin-bottom:2px;">
                                <span class="data-label">{model_info['score_type']}</span>
                                <span class="data-value" style="color: {score_color}">{model_info['primary_score']:.4f}</span>
                            </div>
                            <div class="progress-bg">
                                <div class="progress-fill" style="width: {score_pct}%; background: {score_color};"></div>
                            </div>
                        </div>
                        <div class="data-row">
                            <span class="data-label">🏷️ Total Classes</span>
                            <span class="data-value">{len(labels)}</span>
                        </div>
                        <div class="classes-box" title="{label_str}">{label_str}</div>
                    </div>
                    
                    <div class="info-card">
                        <div class="card-title">📦 Storage & Hardware</div>
                        <div class="data-row">
                            <span class="data-label">File Size</span>
                            <span class="data-value">{model_info.get('model_size_mb', 0):.2f} MB</span>
                        </div>
                        <div class="data-row">
                            <span class="data-label">VRAM Usage</span>
                            <span class="data-value">{model_info.get('vram_gb', 0):.2f} GB</span>
                        </div>
                        <div class="data-row">
                            <span class="data-label">Trained Date</span>
                            <span class="data-value">{trained_at_formatted}</span>
                        </div>
                        <div style="margin-top:10px; font-size:0.75rem; color:#6b7280;">
                            Path: ...{model_path[-30:]}
                        </div>
                    </div>
                    
                    <div class="info-card">
                        <div class="card-title">📈 Key Metrics</div>
                        {''.join([
                            f'<div class="data-row"><span class="data-label">{k}</span><span class="data-value">{v if isinstance(v, str) else f"{v:.4f}"}</span></div>' 
                            for k, v in list(metrics.items())[:6]
                        ])}
                    </div>
                </div>
            </div>
            """
            return html
            
        except Exception as e:
            return f"<p>Error loading model details: {str(e)}</p>"
    
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
