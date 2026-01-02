import gradio as gr
import json
from modelDatabase import ModelRegistry
from datetime import datetime

def create_tab(app):
    with gr.Tab("🎱 Predict Model") as tab:
        # UI components only
        gr.Markdown("## Predict Model")
        gr.Markdown("Select a CVAT project, then choose a trained model to run inference.")
        
        with gr.Column():
            # 1. Project Selection
            registry = ModelRegistry('model_registry.db')
            projects = registry.get_unique_projects()
            
            # Format project choices: "Project 126 (3 models)"
            project_choices = []
            for p in projects:
                choice_label = f"Project {p['cvat_project_id']} ({p['model_count']} models)"
                project_choices.append((choice_label, p['cvat_project_id']))
            
            initial_project_id = project_choices[0][1] if project_choices else None
            
            # 2. Model Selection (filtered by project)
            # Get initial models for first project
            initial_models = registry.list_models(cvat_project_id=initial_project_id) if initial_project_id else []
            model_choices = []
            for m in initial_models:
                choice_label = f"{m['name']} {m['version']} - {m['primary_score_type']}: {m['primary_score']:.2f} - {m['task'].title()}"
                model_choices.append((choice_label, m['id']))
            
            initial_model_id = model_choices[0][1] if model_choices else None

            with gr.Row():
                project_dropdown = gr.Dropdown(
                    label="1. Select CVAT Project",
                    choices=project_choices,
                    value=initial_project_id,
                    interactive=True,
                    elem_id="project_dropdown"
                )
                model_dropdown = gr.Dropdown(
                    label="2. Select Trained Model",
                    choices=model_choices,
                    value=initial_model_id,
                    interactive=True,
                    elem_id="model_dropdown"
                )
            
            with gr.Accordion("📊 Model Details", open=False):
                model_details_html = gr.HTML(
                    value="<p>Select a model to view details</p>"
                )
            
            show_plots_checkbox = gr.Checkbox(
                value=False,
                label="Show Model Training Analysis"
            )

            with gr.Group(visible=False) as plots_group:
                model_plots_gallery = gr.Gallery(
                    label="Model Training Analysis", 
                    show_label=True, 
                    elem_id="model_plots",
                    columns=[4],
                    rows=[1],
                    height=200,
                    allow_preview=True,
                    object_fit="contain",
                    interactive=False
                )
                
            gr.HTML("<div style='margin: 20px 0;'></div>")

            # 3. Prediction Interface
            gr.Markdown("### 📂 Select Test Image")
            test_gallery = gr.Gallery(
                label="Test Images", 
                show_label=False, 
                elem_id="test_gallery",
                columns=[4],
                rows=[1],
                height=150,
                allow_preview=True,
                interactive=True
            )
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🖼️ Run Prediction")
                    input_img = gr.Image(
                        label="Input Image", 
                        type="pil",
                        height=400
                    )
                        
                    conf_slider = gr.Slider(
                        minimum=0.01, maximum=1.0, value=0.25, 
                        step=0.01, label="Confidence Threshold"
                    )
                    
                    iou_slider = gr.Slider(
                        minimum=0.01, maximum=1.0, value=0.45, 
                        step=0.01, label="IOU Threshold"
                    )
                        
                    predict_btn = gr.Button("🚀 Predict Image", variant="primary", elem_id="btn")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Prediction Result")
                    output_img = gr.Image(label="Prediction Result", type="pil", height=400)
                    result_details = gr.Code(label="Detection Details", language="json", elem_id="detection_details_code", lines=10)
    
    return {
        "tab": tab,
        "project_dropdown": project_dropdown,
        "model_dropdown": model_dropdown,
        "model_details_html": model_details_html,
        "show_plots_checkbox": show_plots_checkbox,
        "plots_group": plots_group,
        "model_plots_gallery": model_plots_gallery,
        "test_gallery": test_gallery,
        "input_img": input_img,
        "conf_slider": conf_slider,
        "iou_slider": iou_slider,
        "predict_btn": predict_btn,
        "output_img": output_img,
        "result_details": result_details
    }

def setup_events(app, components, all_components):
    c = components
    
    # Internal logic
    def on_predict_tab_select():
        """Refreshes the project dropdown when tab is selected."""
        app.cleanup_preview()
        registry = ModelRegistry('model_registry.db')
        projects = registry.get_unique_projects()
        
        project_choices = []
        for p in projects:
            choice_label = f"Project {p['cvat_project_id']} ({p['model_count']} models)"
            project_choices.append((choice_label, p['cvat_project_id']))
        
        return gr.update(choices=project_choices, value=None)

    def on_project_change(project_id):
        """Load models for selected project."""
        if not project_id:
            return gr.update(choices=[], value=None), "<p>Select a project first</p>", gr.update(value=[]), gr.update(value=[])
        
        registry = ModelRegistry('model_registry.db')
        models = registry.list_models(cvat_project_id=project_id)
        
        model_choices = []
        for m in models:
            choice_label = f"{m['name']} {m['version']} - {m['primary_score_type']}: {m['primary_score']:.2f} - {m['task'].title()}"
            model_choices.append((choice_label, m['id']))
        
        return (
            gr.update(choices=model_choices, value=None),
            "<p>Select a model to view details</p>",
            gr.update(value=[]),
            gr.update(value=[])
        )

    def format_model_details(model_info):
        """Format model metadata as HTML."""
        if not model_info:
            return "<p>Select a model to view details</p>"
        
        labels = json.loads(model_info['labels']) if isinstance(model_info['labels'], str) else model_info['labels']
        metrics = json.loads(model_info['metrics']) if isinstance(model_info['metrics'], str) else model_info['metrics']
        
        # Format class labels - each on new line
        label_str = '\n'.join([f"{k}: {v}" for k, v in labels.items()]) if isinstance(labels, dict) else str(labels)
        
       # Calculate score percentage for the progress bar (0 to 100)
        # Calculate score percentage (0-100)
        score_pct = model_info.get('primary_score', 0) * 100
        # Colors remain the same for the bar, as they pop well on black
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

        html = f"""
        <style>
            /* Container */
            .model-dashboard {{
                font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
                color: #e5e7eb; /* Light gray text for general body */
                max-width: 100%;
            }}
            
            /* Header Section */
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
                color: #ffffff; /* Pure white title */
            }}
            .version-badge {{ 
                background-color: #312e81; /* Dark Indigo */
                color: #a5b4fc; /* Light Indigo text */
                border: 1px solid #4338ca;
                padding: 4px 10px; 
                border-radius: 99px; 
                font-size: 0.85rem; 
                font-weight: 600;
                text-transform: uppercase;
            }}

            /* Grid Layout */
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 16px;
            }}

            /* DARK CARD STYLE */
            .info-card {{
                background: #1f2937; /* Dark Charcoal Background */
                border: 1px solid #374151; /* Subtle dark border */
                border-radius: 12px;
                padding: 16px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5); /* Heavier shadow for depth */
            }}
            
            .card-title {{
                font-size: 0.95rem; 
                font-weight: 600; 
                color: #9ca3af; /* Muted gray for subtitles */
                margin-bottom: 12px; 
                text-transform: uppercase; 
                letter-spacing: 0.05em;
                border-bottom: 1px solid #374151;
                padding-bottom: 8px;
            }}

            /* Data Rows */
            .data-row {{
                display: flex; 
                justify-content: space-between;
                margin-bottom: 8px; 
                font-size: 0.9rem;
            }}
            .data-label {{ color: #d1d5db; }} /* Light gray label */
            .data-value {{ font-weight: 500; color: #ffffff; text-align: right; }} /* White value */

            /* Visual Elements */
            .progress-bg {{
                background: #374151; /* Dark track for progress bar */
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
                box-shadow: 0 0 8px {score_color}; /* Glow effect on the bar */
            }}
            
            .classes-box {{
                background: #111827; /* Very dark box for classes */
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
                <h3 class="model-title">{model_info['name']}</h3>
                <span class="version-badge">{model_info['version']}</span>
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
                            <span class="data-label">{model_info['primary_score_type']}</span>
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
                        <span class="data-value">{model_info.get('file_size_mb', 0):.2f} MB</span>
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
                        ID: {model_info.get('cvat_project_id', 'N/A')} <br>
                        Path: {model_info['storage_path'][-25:]} 
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

    def on_model_change(model_id):
        """Load model details and related data when selected."""
        if not model_id:
            return "<p>Select a model</p>", gr.update(value=[]), gr.update(value=[]), gr.update(visible=True), gr.update(visible=True)
        
        # Get model from database
        registry = ModelRegistry('model_registry.db')
        model_info = registry.get_model(model_id)
        
        if not model_info:
            return "<p>Model not found</p>", gr.update(value=[]), gr.update(value=[]), gr.update(visible=True), gr.update(visible=True)
        
        # Format details HTML
        details_html = format_model_details(model_info)
        
        # Get test images from project
        cvat_project_id = model_info['cvat_project_id']
        project_name = f"Project_{cvat_project_id}"
        test_imgs = app.get_test_images(project_name) if hasattr(app, 'get_test_images') else []
        
        # Get training plots
        plots = app.get_model_plots(model_info['storage_path']) if hasattr(app, 'get_model_plots') else []
        
        # Detect if it's a classification model
        is_classification = model_info['task'] == 'classify'
        
        return (
            details_html,
            gr.update(value=test_imgs),
            gr.update(value=plots),
            gr.update(visible=not is_classification),  # conf_slider
            gr.update(visible=not is_classification)   # iou_slider
        )

    def on_predict(model_id, img, conf, iou):
        """Run prediction using selected model."""
        if not model_id or not img:
            return None, json.dumps({"error": "Please select a model and image"})
        
        # Get model path from database
        registry = ModelRegistry('model_registry.db')
        model_info = registry.get_model(model_id)
        
        if not model_info:
            return None, json.dumps({"error": "Model not found"})
        
        # Use app's prediction function with model path
        return app.predict_with_model_path(model_info['storage_path'], img, conf, iou)

    def toggle_plots_visibility(checked):
        return gr.update(visible=checked)
    
    def on_gallery_select(evt: gr.SelectData):
        """Handle gallery selection to load image into input"""
        return evt.value["image"]["path"]

    # --- Event Handlers ---
    c["tab"].select(
        fn=on_predict_tab_select, 
        outputs=[c["project_dropdown"]]
    )
    
    c["project_dropdown"].change(
        fn=on_project_change,
        inputs=[c["project_dropdown"]],
        outputs=[c["model_dropdown"], c["model_details_html"], c["test_gallery"], c["model_plots_gallery"]]
    )
    
    c["model_dropdown"].change(
        fn=on_model_change,
        inputs=[c["model_dropdown"]],
        outputs=[c["model_details_html"], c["test_gallery"], c["model_plots_gallery"], c["conf_slider"], c["iou_slider"]]
    )
    
    c["predict_btn"].click(
        fn=on_predict,
        inputs=[c["model_dropdown"], c["input_img"], c["conf_slider"], c["iou_slider"]],
        outputs=[c["output_img"], c["result_details"]]
    )
    
    c["test_gallery"].select(
        fn=on_gallery_select,
        outputs=[c["input_img"]]
    )
    
    c["show_plots_checkbox"].change(
        fn=toggle_plots_visibility, 
        inputs=c["show_plots_checkbox"],
        outputs=c["plots_group"]
    )
