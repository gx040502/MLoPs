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
        result = app.remove_dataset(dataset_name)
        updated_datasets = refresh_datasets_state()
        datasets_html = app.create_dataset_html()
        return result, updated_datasets, datasets_html

    def upload_and_refresh(zip_file) -> tuple:
        """Handle upload and refresh datasets"""
        upload_result = app.upload_dataset_by_zip(zip_file)
        updated_datasets = refresh_datasets_state()
        return upload_result, updated_datasets

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
        with gr.Tab("📤 Upload Dataset") as upload_tab:
            upload_tab.select(fn=lambda: app.cleanup_preview(), outputs=None)
            
            gr.Markdown("## Upload New Dataset")
            gr.Markdown("Upload a ZIP file containing your dataset. It will be extracted and added to the available datasets.")
            
            with gr.Column():
                zip_file_input = gr.File(
                    label="Upload ZIP File", 
                    file_count="single", 
                    file_types=[".zip"], 
                    type="filepath"
                )
                
                upload_btn = gr.Button("📤 Upload & Extract", variant="primary", size="lg",elem_id="btn")
                
                gr.HTML("<div style='margin-top: 15px;'></div>")
                
                upload_status = gr.Textbox(
                    label="Upload Status", 
                    interactive=False, 
                    lines=10,
                    value="Ready to upload..."
                )
        # HOME/DATASETS TAB
        with gr.Tab("📊 Home - Datasets") as datasets_tab:
            datasets_tab.select(fn=lambda: app.cleanup_preview(), outputs=None)

            with gr.Column():
                gr.Markdown("## Available Datasets")
            
                # Datasets display
                datasets_display = gr.HTML(
                    value=app.create_dataset_html(),
                    label="Datasets"
                )
                
                # Status display
                status_output = gr.Textbox(
                    label="Selected Dataset", 
                    interactive=False, 
                    lines=2,
                    value="Ready"
                )

                # Dataset selection
                with gr.Row():
                    dataset_dropdown = gr.Dropdown(
                        choices=[d["name"] for d in refresh_datasets_state()],
                        label="Select Dataset",
                        interactive=True
                    )
                
                # Video Frame Extraction Configuration
                with gr.Group(visible=False) as video_config_group:
                    gr.Markdown("### 🎥 Video Processing Enabled")
                    gr.Markdown("This dataset contains video files. Extracted frames will be combined with existing images into a **temporary dataset** for annotation.")
                    video_dataframe = gr.Dataframe(
                        headers=["Video Name", "Duration", "Extraction Interval"],
                        datatype=["str", "str", "str"],
                        col_count=(3, "fixed"),
                        type="pandas",
                        interactive=True,
                        label="Configure Frame Extraction (e.g., '1s', '5s', '1m')"
                    )

                with gr.Row():
                    with gr.Column():
                        annotate_btn = gr.Button("📂 Annotate", variant="primary",elem_id="btn")
                        delete_btn = gr.Button("🗑️ Delete Selected Dataset", variant="secondary",elem_id="del_btn")
                

        # UPLOAD TAB
        
        with gr.Tab("🧠 Train Model") as train_tab:
            train_tab.select(fn=lambda: app.cleanup_preview(), outputs=None)
            
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
                    lines=5,
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
                        height="auto",
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
                with gr.Row():
                    # Left Column: Input
                    with gr.Column():

                        gr.Markdown("### 📂 Select Test Image")
                        test_gallery = gr.Gallery(
                            label="Test Images", 
                            show_label=False, 
                            elem_id="test_gallery",
                            columns=[1],
                            rows=[1],
                            height="auto",
                            allow_preview=False,
                            interactive=True
                        )
                        gr.Markdown("### 🖼️ Run Prediction")
                        input_img = gr.Image(label="Input Image", type="pil")
                        
                        conf_slider = gr.Slider(
                            minimum=0.01, maximum=1.0, value=0.25, 
                            step=0.01, label="Confidence Threshold"
                        )
                        
                        iou_slider = gr.Slider(
                            minimum=0.01, maximum=1.0, value=0.45, 
                            step=0.01, label="IOU Threshold"
                        )
                        
                        predict_btn = gr.Button("🚀 Predict Image", variant="primary", elem_id="btn")

                    # Right Column: Output
                    with gr.Column():
                        output_img = gr.Image(label="Prediction Result", type="pil")
                        result_details = gr.Code(label="Detection Details", language="json")

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
        html = app.create_dataset_html()
        return gr.update(choices=choices, value=None), html

    def update_formatted_dataset_dropdown_and_display(datasets):
        """Update both dropdown choices and HTML display"""
        choices = [d["name"] for d in datasets]
        return gr.update(choices=choices, value=None)
    
    def refresh_all_components():
        """Refresh all dataset-related components"""
        updated_datasets = refresh_datasets_state()
        dropdown_update, html_update = update_dataset_dropdown_and_display(updated_datasets)
        
        # Refresh CVAT tasks as well
        # Note: We assume project_id is None since the project dropdown is hidden/unused currently
        cvat_tasks = app.get_cvat_tasks(cvat_projects_dropdown.value)
        cvat_tasks_update = gr.update(choices=cvat_tasks)
        
        return updated_datasets, dropdown_update, html_update, "Datasets & Tasks refreshed!", cvat_tasks_update

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
             df_data = [[v["Video Name"], v["Duration"], v["Extraction Interval"]] for v in video_files]
             return msg, gr.update(visible=True), df_data
        else:
             return msg, gr.update(visible=False), None

    dataset_dropdown.change(
        fn=on_dataset_select,
        inputs=[dataset_dropdown],
        outputs=[status_output, video_config_group, video_dataframe]
    )

    def on_annotate_click(dataset_name, video_df):
        # Check if we have video data to process
        import pandas as pd
        has_data = False
        if isinstance(video_df, pd.DataFrame):
            has_data = not video_df.empty
        elif isinstance(video_df, list):
            has_data = len(video_df) > 0
            
        if has_data and dataset_name:
            # Check if there are any actual rows (sometimes empty list might pass through)
            # Actually, let's just try extraction. If scan_for_videos found nothing, user won't see the group, 
            # but Gradio passes the hidden state value.
            # We should rely on whether the dataframe is actually populated with valid data.
            
            # Re-scan to double check? No, expensive.
            # Just trust the dataframe input.
            
            # Logic: If dataframe has content, perform extraction.
            # Note: If the group was hidden, dataframe might still have old value? 
            # Gradio usually creates fresh component instance but let's be careful.
            # We can check visibility? No, inputs don't pass visibility.
            
            # Simple check: does the dataframe have rows?
            if len(video_df) > 0:
                 app.extract_frames_from_dataset(dataset_name, video_df)
                 # New dataset is now selected inside app logic.
        
        return None # Proceed to JS redirect

    annotate_btn.click(
        fn=on_annotate_click,
        inputs=[dataset_dropdown, video_dataframe],
        outputs=None
    ).then(
        None, None, None, js="() => window.location.href = '/vlm'"
    )
    
    delete_btn.click(
        fn=lambda dataset_name, datasets: (
            delete_dataset_handler(dataset_name) if dataset_name 
            else ("Please select a dataset first", datasets, app.create_dataset_html())
        ),
        inputs=[dataset_dropdown, datasets_state],
        outputs=[status_output, datasets_state, datasets_display]
    ).then(
        fn=update_dataset_dropdown_and_display,
        inputs=datasets_state,
        outputs=[dataset_dropdown, datasets_display]
    )
    
    upload_btn.click(
        fn=upload_and_refresh,
        inputs=zip_file_input,
        outputs=[upload_status, datasets_state]
    ).then(
        fn=update_dataset_dropdown_and_display,
        inputs=datasets_state,
        outputs=[dataset_dropdown, datasets_display]
    )
    
    # Initialize the interface
    app_interface.load(
        fn=refresh_all_components,
        outputs=[datasets_state, dataset_dropdown, datasets_display, status_output, cvat_tasks_dropdown]
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