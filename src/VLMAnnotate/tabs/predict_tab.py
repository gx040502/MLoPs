import gradio as gr
import json
from src.ModelManager.ModelManager import ModelManager
from src.ModelManager.DBmanager import DBManager
from datetime import datetime
from pathlib import Path
from PIL import Image
from ultralytics import YOLO


def create_tab(app):
    with gr.Tab("🎱 Predict Model") as tab:
        # UI components only
        gr.Markdown("## Predict Model")
        gr.Markdown("Select a CVAT project, then choose a trained model to run inference.")
        
        with gr.Column():
            # 1. Project Selection
            registry = DBManager('database.db')
            model_manager = ModelManager('database.db')
            projects = model_manager.get_trained_projects()
            
            # Format project choices: "Project 126 (3 models)"
            project_choices = []
            for p in projects:
                choice_label = f"{p['name']} ({p['model_count']} models) ID: {p['cvat_project_id']}"
                project_choices.append((choice_label, p['cvat_project_id']))
            
            initial_project_id = project_choices[0][1] if project_choices else None
            
            # 2. Model Selection (filtered by project)
            # Get initial models for first project
            initial_models = model_manager.get_models(initial_project_id)
            #initial_models = registry.list_models(cvat_project_id=initial_project_id) if initial_project_id else []
            model_choices = []
            for m in initial_models:
                choice_label = f"{m['name']} {m['version']} - {m['task'].title()}: {m['primary_score']:.2f}"
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
                
            gr.HTML("""
                <div style="text-align: center; margin: 15px 0;">
                    <div style="
                        height: 1px; 
                        width: 100%;
                        max-width: 100%;
                        margin: 0 auto;
                        background: linear-gradient(90deg, transparent, #6366f1, transparent); 
                        box-shadow: 0 0 10px rgba(99, 102, 241, 0.5);">
                    </div>
                </div>
            """)

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
                    input_file = gr.File(
                        label="Input Images (Batch), Zip (Images only), or Video", 
                        file_types=["image", "video", ".zip"],
                        file_count="multiple",
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
                        
                    predict_btn = gr.Button("🚀 Predict", variant="primary", elem_id="btn")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Prediction Result")
                    # Gallery for batch images
                    output_gallery = gr.Gallery(
                        label="Prediction Results", 
                        show_label=True, 
                        elem_id="output_gallery", 
                        columns=[3], 
                        rows=[2], 
                        height=400, 
                        object_fit="contain"
                    )
                    output_video = gr.Video(label="Prediction Result Video", height=400, visible=False)
                    # File output for Zip results
                    output_file = gr.File(label="Download Results (Zip)", visible=False)
                    
                    with gr.Accordion("📋 Detection Details", open=False):
                        result_details = gr.Code(label="", language="json", elem_id="detection_details_code", lines=10)
    
    return {
        "tab": tab,
        "project_dropdown": project_dropdown,
        "model_dropdown": model_dropdown,
        "model_details_html": model_details_html,
        "show_plots_checkbox": show_plots_checkbox,
        "plots_group": plots_group,
        "model_plots_gallery": model_plots_gallery,
        "test_gallery": test_gallery,
        "test_gallery": test_gallery,
        "input_file": input_file,
        "conf_slider": conf_slider,
        "conf_slider": conf_slider,
        "iou_slider": iou_slider,
        "predict_btn": predict_btn,
        "output_gallery": output_gallery,
        "output_video": output_video,
        "output_file": output_file,
        "result_details": result_details
    }

def setup_events(app, components, all_components):
    c = components
    model_manager = ModelManager('database.db')
    
    # Internal logic
    def on_predict_tab_select():
        """Refreshes the project dropdown when tab is selected."""
        app.cleanup_preview()
        projects = model_manager.get_trained_projects()
        
        project_choices = []
        for p in projects:
            choice_label = f"{p['name']} ({p['model_count']} models) ID: {p['cvat_project_id']}"
            project_choices.append((choice_label, p['cvat_project_id']))
        
        return gr.update(choices=project_choices, value=None)

    def on_project_change(project_id):
        """Load models for selected project."""
        if not project_id:
            return gr.update(choices=[], value=None), "<p>Select a project first</p>", gr.update(value=[]), gr.update(value=[])
        
        # registry = DBManager('database.db') # Variable already initialized in create_tab but not here
        registry = DBManager('database.db')
        models = registry.list_models(cvat_project_id=project_id)
        
        model_choices = []
        for m in models:
            choice_label = f"{m['name']} {m['version']} - {m['task'].title()}: {m['primary_score']:.2f}"
            model_choices.append((choice_label, m['id']))
        
        return (
            gr.update(choices=model_choices, value=None),
            "<p>Select a model to view details</p>",
            gr.update(value=[]),
            gr.update(value=[])
        )

    def on_model_change(model_id):
        """Load model details and related data when selected."""
        if not model_id:
            return "<p>Select a model</p>", gr.update(value=[]), gr.update(value=[]), gr.update(visible=True), gr.update(visible=True)
        
        # Get model from database
        model_info = model_manager.get_model(model_id)
        
        if not model_info:
            return "<p>Model not found</p>", gr.update(value=[]), gr.update(value=[]), gr.update(visible=True), gr.update(visible=True)
        
        # Format details HTML
        model_trained_args=model_manager.get_model_trained_args(model_info['storage_path'])
        details_html, _ = model_manager.display_model_details(model_id,model_trained_args)
        
        # Get test images from project
        cvat_project_id = model_info['cvat_project_id']
        test_imgs = model_manager.get_test_images(cvat_project_id)
        
        # Get training plots
        plots = app.get_model_plots(model_info['storage_path'])
        
        # Detect if it's a classification model
        is_classification = model_info['task'] == 'classify'
        
        return (
            details_html,
            gr.update(value=test_imgs),
            gr.update(value=plots),
            gr.update(visible=not is_classification),  # conf_slider
            gr.update(visible=not is_classification)   # iou_slider
        )

    def on_predict(model_id, input_files, conf, iou):
        """Run prediction using selected model."""
        # input_files is now a list of file paths (from gr.File(file_count="multiple"))
        # But if user uploads one file, it might be a single string if type="filepath"? 
        # Gradio File component returns a list of file objects or temp paths.
        # Wait, type defaults to 'filepath' which for multiple is a list of strings? Let's assume list of paths.
        
        if not model_id or not input_files:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), json.dumps({"error": "Please select a model and upload media"})
        
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
            
        registry = DBManager('database.db')
        model_manager = ModelManager('database.db')
        model_info = registry.get_model(model_id)
        
        if not model_info:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), json.dumps({"error": "Model not found"})

        model = model_manager.load_model(model_id)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        import zipfile
        import tempfile
        import shutil
        import os
        import cv2
        
        # --- Logic Branching ---
        
        # 1. Single Zip File -> Process Images inside -> Return Zip
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
                     
                 # Process images
                 processed_count = 0
                 for root, dirs, files in os.walk(extract_dir):
                     for file in files:
                         if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                             img_path = os.path.join(root, file)
                             # Calculate relative path to maintain structure if needed, or just flat
                             # For simplicity, flattening output or keeping filenames
                             try:
                                 image = Image.open(img_path)
                                 res_img, _ = model_manager.inference_image(image, model, conf, iou)
                                 # res_img is a PIL Image or numpy array?
                                 # Manager returns annotated_img (numpy array from plot()) usually, let's check.
                                 # If it is numpy, convert to BGR for cv2 save
                                 save_path = os.path.join(output_dir, file)
                                 
                                 if isinstance(res_img, Image.Image):
                                     res_img.save(save_path)
                                 else:
                                     # Expecting numpy array RGB/BGR?
                                     # Ultralytics plot() returns numpy array (BGR usually? No RGB). 
                                     # OpenCV expects BGR. 
                                     # Let's save safely using PIL if possible or cv2
                                     if hasattr(res_img, 'shape'):
                                         # Convert RGB to BGR for opencv if it came from PIL-like plot
                                         # Ultralytics plot() is BGR or RGB? usually RGB for display.
                                         # Let's assume RGB and convert to BGR for cv2.imwrite
                                         res_img_bgr = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
                                         cv2.imwrite(save_path, res_img_bgr)
                                 processed_count += 1
                             except Exception as e:
                                 print(f"Failed to process {file}: {e}")
                 
                 if processed_count == 0:
                     return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), json.dumps({"error": "No valid images found in zip"})

                 # Zip output
                 output_zip = os.path.join(temp_dir, f"predictions_{Path(zip_path).name}")
                 shutil.make_archive(output_zip.replace('.zip', ''), 'zip', output_dir)
                 
                 return gr.update(visible=False), gr.update(visible=False), gr.update(value=output_zip, visible=True), json.dumps({"info": f"Processed {processed_count} images from zip"})
                 
             except Exception as e:
                 return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), json.dumps({"error": str(e)})

        # 2. Single Video File -> Process Video -> Return Video
        if len(input_files) == 1 and Path(input_files[0]).suffix.lower() in video_extensions:
            video_path = input_files[0]
            try:
                output_path, details = model_manager.inference_video(video_path, model, conf, iou)
                if not output_path:
                    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), details
                return gr.update(visible=False), gr.update(value=output_path, visible=True), gr.update(visible=False), details
            except Exception as e:
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), json.dumps({"error": str(e)})

        # 3. Multiple Images (Batch) -> Process All -> Return Gallery
        # Also handles single image
        gallery_results = []
        details_list = []
        
        for file_path in input_files:
            try:
                # Basic check for image extension
                if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    continue
                    
                image = Image.open(file_path)
                output_image, det = model_manager.inference_image(image, model, conf, iou)
                
                # output_image might be numpy array (RGB) from plot()
                # Gallery accepts numpy arrays (RGB)
                gallery_results.append(output_image)
                # Parse details just for summary? Or keep last one?
                # details_list.append(json.loads(det) if det else {})
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if not gallery_results:
             return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), json.dumps({"error": "No valid images processed"})
             
        # Return Gallery
        return gr.update(value=gallery_results, visible=True), gr.update(visible=False), gr.update(visible=False), json.dumps({"info": f"Processed {len(gallery_results)} images"}, indent=2)

    def toggle_plots_visibility(checked):
        return gr.update(visible=checked)
    
    def on_gallery_select(evt: gr.SelectData):
        """Handle gallery selection to load image into input"""
        # Return a list because input_file has file_count="multiple"
        return [evt.value["image"]["path"]]

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
        inputs=[c["model_dropdown"], c["input_file"], c["conf_slider"], c["iou_slider"]],
        outputs=[c["output_gallery"], c["output_video"], c["output_file"], c["result_details"]]
    )
    
    c["test_gallery"].select(
        fn=on_gallery_select,
        outputs=[c["input_file"]]
    )
    
    c["show_plots_checkbox"].change(
        fn=toggle_plots_visibility, 
        inputs=c["show_plots_checkbox"],
        outputs=c["plots_group"]
    )
