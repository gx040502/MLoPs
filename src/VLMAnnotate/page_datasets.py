import gradio as gr
import os
import json
import random
import time
from typing import List, Dict, Any


def load_dataset_interface(app_interface, app):
    """Load the dataset management interface"""
    
    def refresh_datasets_state():
        """Get current datasets and return as state"""
        return app.get_all_datasets()

    def refresh_formatted_datasets_state():
        """Get current formatted datasets and return as state"""
        return app.get_all_formatted_datasets()

    def select_dataset(dataset_name: str) -> str:
        success, message = app.select_dataset(dataset_name)
        return message

    def delete_dataset_handler(dataset_name: str) -> tuple:
        """Handle dataset deletion and return updated state"""
        success, message = app.remove_dataset(dataset_name)
        updated_datasets = refresh_datasets_state()
        return message, updated_datasets

    def upload_and_refresh(zip_file) -> tuple:
        """Handle upload and refresh datasets"""
        success, message = app.upload_dataset_by_zip(zip_file)
        if success:
            message = "Successfully uploaded"
        updated_datasets = refresh_datasets_state()
        return message, updated_datasets

    # Global state to store datasets
    datasets_state = gr.State(value=refresh_datasets_state())
    
    gr.Markdown(
        """
        <div class="page-header">
            <h1>🗂️ Data Management System</h1>
            <p>Upload, manage, and organize your datasets</p>
        </div>
        """, 
        elem_classes=["page-header"]
    )

    # Custom CSS for the button
    gr.HTML("""
        <style>
        #btn {
            background: linear-gradient(45deg, #11998e, #38ef7d);
            border: none;
            color: white !important;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        #del_btn {
            background: linear-gradient(45deg, #FF416C, #FF4B2B);
            border: none;
            color: white !important;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        #del_btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        }
        #btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        }
        /* Global Slider Styling */
        input[type=range] {
            accent-color: #667eea !important; 
            filter: hue-rotate(240deg);
        }
        /* For Firefox */
        input[type=range]::-moz-range-thumb {
            background-color: #667eea !important;
        }
        </style>
    """)
    
    with gr.Tabs() as tabs:
        with gr.Tab("Upload and Select") as upload_select_tab:

           
            with gr.Row():
                # Left Column: Upload
                with gr.Column(scale=1):
                    gr.Markdown("## Upload New Dataset")
                    
                    zip_file_input = gr.File(
                        label="Upload ZIP File", 
                        file_count="single", 
                        file_types=[".zip"], 
                        type="filepath",
                        height=207
                    )
                    
                    upload_btn = gr.Button("📤 Upload & Extract", variant="primary", size="lg",elem_id="btn")
                    
                    upload_status = gr.Markdown(
                        "Ready to upload..."
                    )

                # Right Column: Management
                with gr.Column(scale=1):
                    gr.Markdown("## Available Datasets")
                    
                    dataset_dropdown = gr.Dropdown(
                        choices=[d["name"] for d in refresh_datasets_state()],
                        label="Select Dataset",
                        interactive=True
                    )
                    
                    status_output = gr.Textbox(
                        label="Selected Dataset", 
                        interactive=False, 
                        lines=2,
                        value="Ready"
                    )
                    
                    with gr.Row():
                        delete_btn = gr.Button("🗑️ Delete Selected Dataset", variant="secondary",elem_id="del_btn")
                        annotate_btn = gr.Button("📂 Annotate", variant="primary",elem_id="btn")
                    
                    delete_status = gr.Markdown(visible=True)
                        
                # Video Frame Extraction Configuration
            with gr.Group(visible=False) as video_config_group:
                gr.Markdown("### 🎥 Video Processing Enabled")
                gr.Markdown("This dataset contains video files. Extracted frames will be combined with existing images into a **temporary dataset** for annotation.")
                interval_slider = gr.Slider(
                    minimum=0.0, maximum=60.0, value=1.0, step=0.1, 
                    label="Extraction Interval (seconds)"
                )
                
                video_dataframe = gr.Dataframe(
                    headers=["Video Name", "Duration"],
                    datatype=["str", "str"],
                    col_count=(2, "fixed"),
                    type="pandas",
                    interactive=False,
                    label="Videos to Process"
                )

        # VLM ANNOTATION TAB
        
        with gr.Tab("VLM Annotation", id="vlm_tab") as vlm_tab:
            
            # --- VLM Code Integration ---
            gr.Markdown("## AI Agent to annotate data")
            
            def load_selected_img():
                if app.selected_dataset:
                    dataset_img_path = app.selected_dataset_1st_img_path
                    if os.path.exists(dataset_img_path) and os.path.splitext(dataset_img_path)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                        return dataset_img_path
                return None # No default fallback here to keep it clean, or use placeholder if preferred
            
            def refresh_vlm_dropdown():
                # Refresh dataset list and return update
                all_datasets = app.get_all_datasets()
                choices = [d["name"] for d in all_datasets]
                # Keep currently selected value if valid, or just current app selection
                val = app.selected_dataset if app.selected_dataset else None
                return gr.update(choices=choices, value=val)

            def get_dataset_images_for_gallery():
                if not app.selected_dataset:
                    return []
                success, dataset = app.get_dataset_by_name(app.selected_dataset)
                if not success: return []
                
                path = dataset.get("path")
                if not path or not os.path.exists(path): return []
                
                images = []
                valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
                for f in sorted(os.listdir(path)):
                    if f.lower().endswith(valid_ext):
                        images.append(os.path.join(path, f))
                return images

          
            with gr.Row():
                with gr.Column(scale=1):
                    # --- Dataset Reselection Logic ---
                    gr.Markdown("### 🗂️ Change Dataset")
                    vlm_dataset_dropdown = gr.Dropdown(
                        choices=[d["name"] for d in app.config.get("datasets", [])],
                        label="Select Dataset",
                        value=app.selected_dataset if app.selected_dataset else None,
                        interactive=True,
                        allow_custom_value=True
                    )

                    # Video Config Group (initially hidden)
                    with gr.Group(visible=False) as vlm_video_group:
                        gr.Markdown("#### 🎥 Video Processing Required")
                        
                        with gr.Row():
                            vlm_interval_slider = gr.Slider(
                                minimum=0.0, maximum=60.0, value=1.0, step=0.1, 
                                label="Extraction Interval (seconds)"
                            )
                            
                        vlm_video_dataframe = gr.Dataframe(
                            headers=["Video Name", "Duration"],
                            datatype=["str", "str"],
                            col_count=(2, "fixed"),
                            type="pandas",
                            interactive=False,
                            label="Videos to Process"
                        )
                        vlm_process_btn = gr.Button("⚙️ Process & Load", variant="secondary",elem_id="btn")
                
                    with gr.Column(scale=1):
                        gr.Markdown("### 🖼️ Dataset Gallery")
                        vlm_gallery = gr.Gallery(
                            label="Select Image from Dataset",
                            show_label=False,
                            columns=[4],
                            rows=[1],
                            height= 150, # Managed by CSS
                            allow_preview=True,
                            interactive=True,
                        )
                
                
                
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📸 Sample Image")
                    vlm_image_input = gr.Image(
                        type="pil",
                        label="Upload Image",
                        height=400,
                    )
                        
                    vlm_text_input = gr.Textbox(
                        value="Black Resistor. Wires. Chips.",
                        label="Text Prompt",
                        placeholder="Describe objects to detect (e.g., 'a person. a car. a dog.')",
                        lines=2
                    )

                    # ---------------------------------
                        
                    vlm_confidence_slider = gr.Slider(
                        minimum=0.01, maximum=1.0, value=0.15, step=0.01, label="Confidence Threshold"
                    )

                    detect_btn = gr.Button("🔍 Detect Objects", variant="primary", size="lg", elem_id="btn")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", elem_id="del_btn",visible=False)
                        
                    
                
                with gr.Column(scale=1):
                        
                    gr.Markdown("### 📊 Output")
                    prompt_output = gr.Textbox(
                        label="Parameters Used ", 
                        lines=2, 
                        show_copy_button=False,
                        visible=False
                    )
                    vlm_output_image = gr.Image(label="Detection Results", height=400)
                    inference_btn = gr.Button("Inference Dataset", variant="primary", elem_id="btn")
                    inference_output = gr.HTML(
                        padding=False,
                        label="Reference Output",
                        value=(
                            "<div style='padding: var(--size-2); "
                            "border: 1px solid var(--block-border-color); "
                            "background: var(--input-background-fill); "
                            "border-radius: var(--container-radius); "
                            "min-height: 80px; "
                            "width: 100%; "
                            "box-sizing: border-box; "  
                            "color: var(--body-text-color);'>"
                            "Ready for inference on dataset</div>"
                        )
                    )

                    # CVAT Integration Elements
                    cvat_project_dropdown = gr.Dropdown(
                        label="Assign to CVAT Project", 
                        choices=app.get_cvat_projects(), 
                        visible=False, 
                        interactive=False,
                        elem_id="cvat_project_dd"
                    )
                    cvat_btn = gr.Button("Create CVAT Task", visible=False, variant="primary")
                    
                    detection_info = gr.Textbox(label="Detection Details", lines=10, show_copy_button=True,visible=False)
                    raw_output = gr.Textbox(label="Raw Results", lines=5, show_copy_button=True,visible=False)

            # Event handlers
            detect_btn.click(
                fn=app.process_image,
                inputs=[vlm_image_input, vlm_text_input, vlm_confidence_slider],
                outputs=[vlm_output_image, detection_info, raw_output]
            ).then(
                fn=lambda prompt, conf: f"Prompt: {prompt}\nConfidence: {conf}",
                inputs=[vlm_text_input, vlm_confidence_slider],
                outputs=[prompt_output]
            )
            
            # --- VLM Dataset Reselection Events ---
            def on_vlm_dataset_select(dataset_name):
                if not dataset_name: return gr.update(visible=False), None
                video_files = app.scan_for_videos(dataset_name)
                has_videos = len(video_files) > 0
                if has_videos:
                     df_data = [[v["Video Name"], v["Duration"]] for v in video_files]
                     return gr.update(visible=True), df_data
                else:
                     app.select_dataset(dataset_name)
                     return gr.update(visible=False), None

            vlm_dataset_dropdown.change(
                fn=on_vlm_dataset_select,
                inputs=[vlm_dataset_dropdown],
                outputs=[vlm_video_group, vlm_video_dataframe]
            ).then(
                fn=load_selected_img, outputs=[vlm_image_input]
            ).then(
                fn=get_dataset_images_for_gallery, outputs=[vlm_gallery]
            )
            
            def on_vlm_process_click(dataset_name, video_df, interval_val):
                success, result = app.extract_frames_from_dataset(
                    dataset_name, video_df, interval_val=interval_val
                )
                
                # Refresh dropdown choices as new temp dataset is created
                all_datasets = app.get_all_datasets()
                choices = [d["name"] for d in all_datasets]
                
                # If success, result is temp_name which is already selected in app
                new_val = result if success else dataset_name
                
                return gr.update(visible=False), gr.update(choices=choices, value=new_val)

            vlm_process_btn.click(
                fn=on_vlm_process_click,
                inputs=[vlm_dataset_dropdown, vlm_video_dataframe, vlm_interval_slider],
                outputs=[vlm_video_group, vlm_dataset_dropdown]
            ).then(
                fn=load_selected_img, outputs=[vlm_image_input]
            ).then(
                fn=get_dataset_images_for_gallery, outputs=[vlm_gallery]
            )

            clear_btn.click(
                lambda: [None, "", 0.15, None, "", ""],
                outputs=[vlm_image_input, vlm_text_input, vlm_confidence_slider, vlm_output_image, detection_info, raw_output]
            )

            inference_btn.click(
                fn=app.inference_dataset,
                inputs=[vlm_text_input, vlm_confidence_slider],
                outputs=[inference_output]
            ).then(
                fn=lambda: [gr.Button(visible=False), gr.Button(visible=True), gr.Dropdown(visible=True)],
                outputs=[inference_btn, cvat_btn, cvat_project_dropdown]
            )

            cvat_btn.click(
                fn=app.create_cvat_task,
                inputs=[cvat_project_dropdown],
                outputs=[inference_output]
            ).then(
                fn=lambda: [gr.Button(visible=True), gr.Button(visible=False), gr.Dropdown(visible=False)],
                outputs=[inference_btn, cvat_btn, cvat_project_dropdown]
            )
            
            vlm_text_input.submit(
                fn=app.process_image,
                inputs=[vlm_image_input, vlm_text_input, vlm_confidence_slider],
                outputs=[vlm_output_image, detection_info, raw_output]
            )
            
            # Tab Select Event: Refresh Dropdown AND Load Image AND Gallery
            vlm_tab.select(
                fn=refresh_vlm_dropdown, 
                outputs=[vlm_dataset_dropdown]
            ).then(
                fn=load_selected_img, 
                outputs=[vlm_image_input]
            ).then(
                fn=get_dataset_images_for_gallery, 
                outputs=[vlm_gallery]
            )

            # Gallery Select Event
            def on_gallery_select(evt: gr.SelectData):
                return evt.value["image"]["path"]

            vlm_gallery.select(
                fn=on_gallery_select,
                outputs=[vlm_image_input]
            )


        # UPLOAD TAB (Original placeholder)
        
        with gr.Tab("🧠 Train Model") as train_tab:

            
            gr.Markdown("## Train New Model")
            gr.Markdown("Train a new model from your CVAT Task")
            
            with gr.Column():
                cvat_projects_dropdown = gr.Dropdown(
                    label="Choose CVAT Projects", 
                    choices=app.get_cvat_projects(), 
                    visible=False, 
                    interactive=False
                )

                def on_tab_select(evt: gr.SelectData):
                    # Clean up preview when switching tabs
                    app.cleanup_preview()

                # Bind cleanup to all tabs that might be switched TO
                # We can bind to the tab object itself


                cvat_tasks_dropdown = gr.Dropdown(
                    label="Choose CVAT Tasks", 
                    choices=app.get_cvat_tasks(cvat_projects_dropdown.value), 
                    visible=True, 
                    interactive=True
                )
                


                train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg", elem_id="btn")
                
                dataset_log = gr.Textbox(
                    label="Dataset Detail Log", 
                    interactive=False, 
                    lines=5,
                    value="Waiting for dataset..."
                )
                
                training_log = gr.Textbox(
                    label="Training Log", 
                    interactive=False, 
                    lines=2,
                    value="Waiting to start..."
                )
                # ETA Components
                eta_output = gr.Markdown("⏳ Estimated Time: Waiting to start...", visible=False)
                current_training_project_name = gr.State(None) 
                
                filename_input = gr.Textbox(
                    label="Custom Output Filename (Optional but this will be the train model name)", 
                    placeholder="e.g. my_dataset.zip",
                    lines=1,
                    visible=False
                )

                gr.HTML("<div style='margin-top: 5px;'></div>")

                gr.Markdown("### ⚙️ Training Configuration")
                with gr.Row():
                    experiments_slider = gr.Slider(
                        minimum=1, 
                        maximum=1000, 
                        value=100, 
                        step=1, 
                        label="Epochs"
                    )
                    
                    imgsz_slider = gr.Slider(
                        minimum=64, 
                        maximum=1280, 
                        value=640, 
                        step=32, 
                        label="Image Size"
                    )
                    
                    model_dropdown = gr.Dropdown(
                        choices=app.get_pretrained_models(),
                        label="Select Pre-trained Model",
                        interactive=True
                    )

                gr.Markdown("#### 📊 Dataset Split Ratios (Must sum approx to 10)")
                with gr.Row():
                    train_ratio = gr.Number(value=7, label="Train Ratio", precision=0)
                    val_ratio = gr.Number(value=2, label="Validation Ratio", precision=0)
                    test_ratio = gr.Number(value=1, label="Test Ratio", precision=0)
                
                manual_aug_checkbox = gr.Checkbox(
                    value=False,
                    label="Manual Adjust Augmentation"
                )
                    
                base_image_state = gr.State(None)
                    
                with gr.Group(visible=False) as aug_settings_group:
                    with gr.Row():
                        with gr.Column():
                            aug_preview_img = gr.Image(label="Augmentation Preview", interactive=False,type="pil")
                            gr.Markdown("### 👁️ Live Preview\nClick below to load a random image from the task to see how augmentations affect it.")
                            load_sample_btn = gr.Button("🎲 Load New Sample", size="sm")

                        with gr.Column():
                            gr.Markdown("#### 🎨 Color Augmentation")
                            with gr.Row():
                                hsv_h = gr.Slider(0.0, 1.0, value=0.015, step=0.001, label="HSV-Hue")
                                hsv_s = gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="HSV-Saturation")
                            with gr.Row():
                                hsv_v = gr.Slider(0.0, 1.0, value=0.4, step=0.01, label="HSV-Value")
                                bgr = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="BGR Flip Prob")

                            gr.Markdown("#### 📐 Geometric Transforms")
                            with gr.Row():
                                degrees = gr.Slider(-180, 180, value=0.0, step=1.0, label="Rotation (+/- deg)")
                                translate = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="Translate (+/- frac)")
                            with gr.Row():
                                scale = gr.Slider(0.0, 2.0, value=0.5, step=0.01, label="Scale (+/- gain)")
                                shear = gr.Slider(-180, 180, value=0.0, step=1.0, label="Shear (+/- deg)")
                            with gr.Row():
                                perspective = gr.Slider(0.0, 0.001, value=0.0, step=0.0001, label="Perspective")
                                flipud = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Flip Up-Down Prob")
                            with gr.Row():
                                fliplr = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Flip Left-Right Prob")

                            gr.Markdown("#### 🔀 Advanced Mixing")
                            with gr.Row():
                                mosaic = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Mosaic Prob")
                                mixup = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Mixup Prob")
                            with gr.Row():
                                cutmix = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Cutmix Prob")
                                copy_paste = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Copy-Paste Prob")

                            gr.Markdown("#### ✨ Image Quality & Effects")
                            with gr.Row():
                                erasing = gr.Slider(0.0, 0.9, value=0.4, step=0.01, label="Erasing %")
                        

                
                def toggle_aug_settings(checkbox_val):
                    return gr.update(visible=checkbox_val)
                
                manual_aug_checkbox.change(fn=toggle_aug_settings, inputs=manual_aug_checkbox, outputs=aug_settings_group)




                def trigger_training(task_id,custom_name, model, epochs, imgsz, manual_aug, 
                                     r_train, r_val, r_test,
                                     h_h, h_s, h_v, bgr_p, 
                                     deg, trans, scl, shr, 
                                     persp, f_ud, f_lr, 
                                     mos, mix, cut, cp, 
                                     ers):
                    # 1. Format/Prepare Data
                    split_ratios = (r_train, r_val, r_test)
                    msg, path = app.process_cvat_task(task_id, custom_name, split_ratios)
                    if "Error" in msg:
                         yield msg, "❌ Format Failed", None
                         return

                    # Inspect Dataset Stats
                    stats = app.inspect_dataset_zip(path)
                    
                    stats_str = "📊 Dataset Stats:\n"
                    if stats.get("status") != "Error":
                        stats_str += f"Valid Images: {stats['images']}\n"
                        stats_str += f"Classes ({stats['classes']}): {stats['class_names']}\n"
                    else:
                        stats_str += f"Error inspecting stats: {stats.get('message')}\n"
                        
                    yield f"{msg}\n\n🚀 Training Started...", stats_str, None


                    # 2. Augmentation Params
                    aug_args = {
                        'hsv_h': h_h, 'hsv_s': h_s, 'hsv_v': h_v, 'bgr': bgr_p,
                        'degrees': deg, 'translate': trans, 'scale': scl, 'shear': shr,
                        'perspective': persp, 'flipud': f_ud, 'fliplr': f_lr,
                        'mosaic': mos, 'mixup': mix, 'cutmix': cut, 'copy_paste': cp,
                        'erasing': ers
                    }
                    
                    # 3. Start Training
                    # Determine Project Name logic to match app.py process_cvat_task
                    # Logic: if custom_name used, project is custom_name (cleaned). Else Task_{id}.
                    
                    project_name = ""
                    if custom_name and custom_name.strip():
                        s_name = custom_name.strip()
                        if s_name.lower().endswith('.zip'): 
                            project_name = s_name[:-4]
                        else:
                            project_name = s_name
                    else:
                        # Default naming: Task_{id}
                        # We need to extract clean ID from "720: TaskName"
                        clean_id = str(task_id).split(':')[0].strip()
                        project_name = f"Task_{clean_id}"
                        
                    yield "🚀 Training Started... ETA should appear shortly.", stats_str, project_name
                    
                    result = app.start_training(path, model, epochs, imgsz, manual_aug, **aug_args)
                    yield result, stats_str, None

                
                def check_training_status(project_name):
                    if not project_name: return gr.update(visible=False)
                    
                    from pathlib import Path
                    import json

                    # Direct path to the status file
                    status_path = Path("Dataset") / project_name / "training_status.json"
                    
                    if not status_path.exists():
                        # Debug info
                        return gr.update(value=f"⏳ Estimated Time: Initializing... (Waiting for {project_name})", visible=True)
                        
                    try:
                        with open(status_path, 'r') as f:
                            data = json.load(f)
                            
                        # Format ETA
                        rem_seconds = data.get("est_time_remaining", 0)
                        total_seconds = data.get("est_total_time", 0)
                        
                        def fmt(s):
                            s = int(s)
                            m, s = divmod(s, 60)
                            return f"{m}m {s}s"
                            
                        return gr.update(value=f"⏳ **Epoch {data['current_epoch']}/{data['total_epochs']}** | Remaining: {fmt(rem_seconds)}", visible=True)
                    except Exception as e:
                        return gr.update(value=f"⏳ Estimated Time: Error reading status ({e})", visible=True)

                # Timer (polls every 2 seconds)
                eta_timer = gr.Timer(2)
                eta_timer.tick(
                    fn=check_training_status, 
                    inputs=[current_training_project_name], 
                    outputs=[eta_output]
                )
                
                train_btn.click(
                    fn=trigger_training,
                    inputs=[
                        cvat_tasks_dropdown, filename_input, model_dropdown, experiments_slider, imgsz_slider, manual_aug_checkbox,
                        train_ratio, val_ratio, test_ratio,
                        hsv_h, hsv_s, hsv_v, bgr,
                        degrees, translate, scale, shear,
                        perspective, flipud, fliplr,
                        mosaic, mixup, cutmix, copy_paste,
                        erasing
                    ],
                    outputs=[training_log, dataset_log, current_training_project_name]
                )
                
                # --- PREVIEW LOGIC ---
                def load_sample_image(task_str):
                    if not task_str: return None, None
                    task_str = str(task_str) # Ensure string
                    task_id = task_str.split(':')[0].strip() if ':' in task_str else task_str
                    # Get list of images (up to 4)
                    images = app.get_random_sample_images(task_id)
                    if not images: return None, None
                    return images, images[0] # State stores list, UI shows 1st image initially

                def update_aug_preview(images, h_h, h_s, h_v, bgr_p, deg, trans, scl, shr, persp, f_ud, f_lr, mos, mix, cut, cp, ers):
                    if not images: return None
                    aug_args = {
                        'hsv_h': h_h, 'hsv_s': h_s, 'hsv_v': h_v, 'bgr': bgr_p,
                        'degrees': deg, 'translate': trans, 'scale': scl, 'shear': shr,
                        'perspective': persp, 'flipud': f_ud, 'fliplr': f_lr,
                        'mosaic': mos, 'mixup': mix, 'cutmix': cut, 'copy_paste': cp,
                        'erasing': ers
                    }
                    return app.preview_augmentation(images, **aug_args)
                
                # Wire events
                load_sample_btn.click(
                    fn=load_sample_image,
                    inputs=[cvat_tasks_dropdown],
                    outputs=[base_image_state, aug_preview_img]
                )
                
                # Update preview when sliders change
                aug_inputs = [
                    base_image_state,
                    hsv_h, hsv_s, hsv_v, bgr,
                    degrees, translate, scale, shear,
                    perspective, flipud, fliplr,
                    mosaic, mixup, cutmix, copy_paste,
                    erasing
                ]
                
                for slider in aug_inputs[1:]:
                    slider.change(fn=update_aug_preview, inputs=aug_inputs, outputs=aug_preview_img)
                for slider in aug_inputs[1:]:
                    slider.change(fn=update_aug_preview, inputs=aug_inputs, outputs=aug_preview_img)
        with gr.Tab("🧠 Predict Model") as predict_tab:
            
            def on_predict_tab_select():
                """Refreshes the project dropdown when tab is selected."""
                app.cleanup_preview()
                projects = app.list_trained_projects()
                return gr.update(choices=projects), gr.update(choices=[], value=None)

            gr.Markdown("## Predict Model")
            gr.Markdown("Select a trained model and run inference on custom images.")
            
            with gr.Column():
                # 1. Model Selection
                with gr.Row():
                    train_project_dd = gr.Dropdown(
                        label="1. Select Project",
                        choices=app.list_trained_projects(),
                        interactive=True,
                        elem_id="train_project_dd"
                    )
                    
                    train_model_dd = gr.Dropdown(
                        label="2. Select Model",
                        choices=[], # Dynamic
                        interactive=True,
                        elem_id="train_model_dd"
                    )
                
                model_details = gr.Textbox(
                    label="Model Details",
                    lines=3,
                    interactive=False
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

                def toggle_plots_visibility(checked):
                    return gr.update(visible=checked)
                
                show_plots_checkbox.change(
                    fn=toggle_plots_visibility, 
                    inputs=show_plots_checkbox,
                    outputs=plots_group
                )
                
                # 2. Prediction Interface
                
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
                        output_img = gr.Image(label="Prediction Result", type="pil",height=400)
                        result_details = gr.Code(label="Detection Details", language="json", elem_id="detection_details_code",lines=10)

                # --- Events for Predict Tab ---
                
                def on_project_change(project_name):
                    # Update model dropdown
                    models = app.list_trained_models(project_name)
                    # Update test images
                    test_imgs = app.get_test_images(project_name)
                    return gr.update(choices=models, value=None), "", gr.update(value=test_imgs)

                def on_model_change(project_name, model_name):
                    # Update details
                    details = app.get_model_details(project_name, model_name)
                    # Update plots
                    plots = app.get_model_plots(project_name, model_name)
                    return details, gr.update(value=plots)

                def on_gallery_select(evt: gr.SelectData):
                    return evt.value["image"]["path"]

                def on_predict(project, model, img, conf, iou):
                    return app.predict_with_model(project, model, img, conf, iou)

                # Wiring
                train_project_dd.change(
                    fn=on_project_change, 
                    inputs=[train_project_dd], 
                    outputs=[train_model_dd, model_details, test_gallery]
                )
                
                train_model_dd.change(
                    fn=on_model_change,
                    inputs=[train_project_dd, train_model_dd],
                    outputs=[model_details, model_plots_gallery]
                )
                
                predict_btn.click(
                    fn=on_predict,
                    inputs=[train_project_dd, train_model_dd, input_img, conf_slider, iou_slider],
                    outputs=[output_img, result_details]
                )
                
                test_gallery.select(
                    fn=on_gallery_select,
                    outputs=[input_img]
                )

                # Connect the refresh function to the tab select event now that components are defined
                predict_tab.select(
                    fn=on_predict_tab_select, 
                    outputs=[train_project_dd, train_model_dd]
                )
            

                
 
        # ABOUT TAB
        with gr.Tab("ℹ️ About") as about_tab:
            about_tab.select(fn=lambda: app.cleanup_preview(), outputs=None)

            gr.Markdown("""
            ## About Dataset Manager
            
            This application helps you manage your datasets efficiently:
            
            ### Features:
            - 📤 **Upload**: Upload ZIP files containing your datasets
            - 📊 **View**: Browse all available datasets with details
            - 🔄 **Refresh**: Real-time updates when datasets change
            - 🗑️ **Delete**: Remove datasets you no longer need
            - 💾 **Persist**: All dataset information is saved in settings.json
            
            ### How to use:
            1. Go to the **Upload** tab to add new datasets
            2. Use the **Home** tab to view and manage existing datasets
            3. Select datasets from the dropdown to view details
            
            ### File Structure:
            - Uploaded datasets are extracted to `unzipped_files/`
            - Dataset metadata is stored in `settings.json`
            - Each dataset entry includes name, path, and upload timestamp
            """)

    # Event handlers
    def update_dataset_dropdown_and_display(datasets):
        """Update both dropdown choices and HTML display"""
        choices = [d["name"] for d in datasets]
        return gr.update(choices=choices, value=None)

    def update_formatted_dataset_dropdown_and_display(datasets):
        """Update both dropdown choices and HTML display"""
        choices = [d["name"] for d in datasets]
        return gr.update(choices=choices, value=None)
    
    def refresh_all_components():
        """Refresh all dataset-related components"""
        updated_datasets = refresh_datasets_state()
        dropdown_update = update_dataset_dropdown_and_display(updated_datasets)
        
        # Refresh CVAT tasks as well
        # Note: We assume project_id is None since the project dropdown is hidden/unused currently
        cvat_tasks = app.get_cvat_tasks(cvat_projects_dropdown.value)
        cvat_tasks_update = gr.update(choices=cvat_tasks)
        
        return updated_datasets, dropdown_update, "Datasets & Tasks refreshed!", cvat_tasks_update

    def refresh_formatted_dropdown():
        """Refresh formatted dataset dropdown"""
        updated_formatted_datasets = refresh_formatted_datasets_state()
        formatted_dropdown_update = update_formatted_dataset_dropdown_and_display(updated_formatted_datasets)
        return formatted_dropdown_update
    
    def on_dataset_select(dataset_name):
        # 1. Select the dataset in app
        success, msg = app.select_dataset(dataset_name) if dataset_name else (False, "Please select a dataset first")
        if not dataset_name:
            return msg, gr.update(visible=False), None
            
        # 2. Check for videos
        video_files = app.scan_for_videos(dataset_name)
        has_videos = len(video_files) > 0
        
        # 3. Prepare config DF data
        # video_files is list of dicts. 
        # DF expects list of [Name, Duration, Interval]
        if has_videos:
             df_data = [[v["Video Name"], v["Duration"]] for v in video_files]
             return msg, gr.update(visible=True), df_data
        else:
             return msg, gr.update(visible=False), None

    dataset_dropdown.change(
        fn=on_dataset_select,
        inputs=[dataset_dropdown],
        outputs=[status_output, video_config_group, video_dataframe]
    )

    def on_annotate_click(dataset_name, video_df, interval_val):
        # 1. Validation
        if not dataset_name:
            return (
                "⚠️ Please select a dataset first", 
                gr.update(), # No tab switch
                gr.update(), # No dropdown update
                gr.update(), # No image update
                gr.update()  # No gallery update
            )
            
        target_dataset = dataset_name
        
        # 2. Check for video processing
        import pandas as pd
        has_data = False
        if isinstance(video_df, pd.DataFrame):
            has_data = not video_df.empty
        elif isinstance(video_df, list):
            has_data = len(video_df) > 0
            
        if has_data:
             # This creates a temp dataset and selects it in app
             success, result = app.extract_frames_from_dataset(dataset_name, video_df, interval_val=interval_val)
             if success:
                 target_dataset = result
        else:
             # Just select the regular dataset
             app.select_dataset(target_dataset)
        
        # 3. Prepare Updates
        # Refresh datasets list in case a temp dataset was just created
        all_datasets = app.get_all_datasets()
        choices = [d["name"] for d in all_datasets]
        
        # Get first image of the target dataset
        # Note: app.selected_dataset is already updated by select_dataset/extract_frames
        img = load_selected_img() 
        gallery_imgs = get_dataset_images_for_gallery()

        return (
            f"✅ Loaded {target_dataset}", 
            gr.update(selected="vlm_tab"), 
            gr.update(choices=choices, value=target_dataset),
            img,
            gallery_imgs
        )

    annotate_btn.click(
        fn=on_annotate_click,
        inputs=[dataset_dropdown, video_dataframe, interval_slider],
        outputs=[status_output, tabs, vlm_dataset_dropdown, vlm_image_input, vlm_gallery]
    )
    
    delete_btn.click(
        fn=lambda dataset_name, datasets: (
            delete_dataset_handler(dataset_name) if dataset_name 
            else ("Please select a dataset first", datasets)
        ),
        inputs=[dataset_dropdown, datasets_state],
        outputs=[delete_status, datasets_state]
    ).then(
        fn=update_dataset_dropdown_and_display,
        inputs=datasets_state,
        outputs=[dataset_dropdown]
    )
    
    upload_btn.click(
        fn=upload_and_refresh,
        inputs=zip_file_input,
        outputs=[upload_status, datasets_state]
    ).then(
        fn=update_dataset_dropdown_and_display,
        inputs=datasets_state,
        outputs=[dataset_dropdown]
    )

    def cleanup_and_refresh_ui():
        """Cleanup temp datasets and refresh all dropdowns"""
        # 1. Cleanup backend
        app.cleanup_temp_datasets()
        
        # 2. Get fresh list
        all_datasets = app.get_all_datasets()
        choices = [d["name"] for d in all_datasets]
        
        # 3. Return updates for both dropdowns
        # Select the first available option if choices exist, else None
        new_val = choices[0] if choices else None
        
        # We also need to tell the app about this selection change implicitly?
        # Ideally we should trigger a selection event, but setting the value here updates the UI.
        
        return (
            gr.update(choices=choices, value=new_val), # For dataset_dropdown (Home)
            gr.update(choices=choices, value=new_val)  # For vlm_dataset_dropdown (VLM)
        )

    # Bind cleanup to all non-VLM tabs
    # We need to target the tab objects defined earlier: 
    # upload_tab, train_tab, predict_tab, about_tab
    
    # Note: We need to ensure we are updating the correct components.
    # dataset_dropdown is in local scope here.
    # vlm_dataset_dropdown is inside load_dataset_interface but we need access to it.
    # Actually vlm_dataset_dropdown is defined deeper in the scope. 
    # To fix this properly, we should define this helper earlier or ensure variable access.
    # However, gradio components are objects, if we can't reach them, we can't update them.

    # REFACTOR STRATEGY: 
    # vlm_dataset_dropdown is defined inside `with gr.Tab("VLM Annotation", ...)` block.
    # We need to make sure we return it or have access to it.
    # In this function `load_dataset_interface`, `vlm_dataset_dropdown` is a local variable 
    # defined around line 196. Since `cleanup_and_refresh_ui` is also inside `load_dataset_interface`,
    # it captures `vlm_dataset_dropdown` via closure. So this is safe.

    common_inputs = [] 
    common_outputs = [dataset_dropdown, vlm_dataset_dropdown]

    upload_select_tab.select(fn=cleanup_and_refresh_ui, inputs=common_inputs, outputs=common_outputs)
    train_tab.select(fn=cleanup_and_refresh_ui, inputs=common_inputs, outputs=common_outputs)
    predict_tab.select(fn=cleanup_and_refresh_ui, inputs=common_inputs, outputs=common_outputs)
    about_tab.select(fn=cleanup_and_refresh_ui, inputs=common_inputs, outputs=common_outputs)
    
    # Initialize the interface
    app_interface.load(
        fn=refresh_all_components,
        outputs=[datasets_state, dataset_dropdown, status_output, cvat_tasks_dropdown]
    )

if __name__ == "__main__":

    from app import APP
    app = APP()

    with gr.Blocks(title="See.AI Agent") as app_interface:
        load_dataset_interface(app)
        gr.Button("Go to VLM", link="/vlm")

    app_interface.launch(
        debug=True,
        share=False,
        server_name="0.0.0.0",
        server_port=6605
    )