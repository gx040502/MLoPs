import gradio as gr
from src.VLMAnnotate.ui_logic import refresh_datasets_state, navigate_to_train_with_dataset

def create_tab(app):
    with gr.Tab("📤 Upload and Select") as tab:
        # ============================================================
        # 1. Upload Raw Dataset Section
        # ============================================================
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
                
                upload_btn = gr.Button("📤 Upload & Extract", variant="primary", size="lg", elem_id="btn",visible=True)
                
                upload_status = gr.Markdown(
                    "Ready to upload..."
                )

            # Right Column: Management
            with gr.Column(scale=1):
                gr.Markdown("## Available Datasets")

                raw_datasets_initial = app.get_all_datasets()
                raw_initial_choices = [d["name"] for d in raw_datasets_initial]
                raw_initial_value = raw_initial_choices[0] if raw_initial_choices else None

                checkbox_formatted = gr.Checkbox(
                    label="Formatted Dataset", 
                    interactive=True,
                    value=False
                )
                dataset_format_dropdown = gr.Dropdown(
                    choices=["COCO 1.0", "Ultralytics YOLO Classification 1.0", "Ultralytics YOLO Detection 1.0", "Ultralytics YOLO Segmentation 1.0"],
                    value="COCO 1.0",
                    label="Dataset Format",
                    interactive=True,
                    visible=False
                )

                dataset_dropdown = gr.Dropdown(
                    choices=raw_initial_choices,
                    value=raw_initial_value,
                    label="Select Dataset",
                    interactive=True,
                    visible=True
                )
                
                status_output = gr.Textbox(
                    label="Selected Dataset", 
                    interactive=False, 
                    lines=2,
                    value="Ready",
                    visible=True
                )
                
                with gr.Row():
                    delete_btn = gr.Button("🗑️ Delete Selected Dataset", variant="secondary", elem_id="del_btn",visible=True)
                    annotate_btn = gr.Button("📂 Annotate", variant="primary", elem_id="btn",visible=True)
                    upload_cvat_btn = gr.Button("📤 Upload to CVAT", variant="primary", size="lg", elem_id="btn",visible=False)
                
                delete_status = gr.Markdown(visible=True)
                upload_cvat_status = gr.Markdown(
                    "Ready to upload..."
                )

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

    return {
        "tab": tab,
        "zip_file_input": zip_file_input,
        "upload_btn": upload_btn,
        "upload_status": upload_status,
        "dataset_dropdown": dataset_dropdown,
        "status_output": status_output,
        "delete_btn": delete_btn,
        "annotate_btn": annotate_btn,
        "delete_status": delete_status,
        "video_config_group": video_config_group,
        "interval_slider": interval_slider,
        "video_dataframe": video_dataframe,
        "checkbox_formatted": checkbox_formatted,
        "dataset_format_dropdown": dataset_format_dropdown,
        "upload_cvat_btn": upload_cvat_btn,
        "upload_cvat_status": upload_cvat_status
    }

def setup_events(app, components, all_components):
    # Unpack specific components needed for this tab
    c = components
    def toggle_button_mode(checked):
        if checked:
            return (
                gr.update(visible=True),  # dataset_format_dropdown
                gr.update(visible=True),  # upload_cvat_btn
                gr.update(visible=False),   # delete_btn
                gr.update(visible=False),   # annotate_btn
                gr.update(visible=False),   # dataset_dropdown
                gr.update(visible=False),   # status_output
                gr.update(visible=False),   # upload_btn
            )
        else:
            return (
                gr.update(visible=False),  # dataset_format_dropdown
                gr.update(visible=False),   # upload_cvat_btn
                gr.update(visible=True),   # delete_btn
                gr.update(visible=True),   # annotate_btn
                gr.update(visible=True),   # dataset_dropdown
                gr.update(visible=True),   # status_output
                gr.update(visible=True),   # upload_btn
            )
    
    def upload_cvat_handler(zip_file, dataset_format):
        """Handle CVAT upload button click"""
        if not zip_file:
            return "❌ Please upload a zip file first"
        
        success, message = app.upload_to_cvat(zip_file, dataset_format)
        return message
    
    # Internal logic functions
    def upload_and_refresh(zip_file) -> tuple: 
        """Handle upload and refresh datasets"""
        success, message = app.upload_dataset_by_zip(zip_file)
        if success:
            message = "Successfully uploaded"
        updated_datasets = refresh_datasets_state(app)
        return message, updated_datasets

    def delete_dataset_handler(dataset_name: str) -> tuple:
        """Handle dataset deletion and return updated state"""
        success, message = app.remove_dataset(dataset_name)
        updated_datasets = refresh_datasets_state(app)
        return message, updated_datasets

    def on_dataset_select(dataset_name):
        # 1. Select the dataset in app
        success, msg = app.select_dataset(dataset_name) if dataset_name else (False, "Please select a dataset first")
        if not dataset_name:
            return msg, gr.update(visible=False), None
            
        # 2. Check for videos
        video_files = app.scan_for_videos(dataset_name)
        has_videos = len(video_files) > 0
        
        # 3. Prepare config DF data
        if has_videos:
             df_data = [[v["Video Name"], v["Duration"]] for v in video_files]
             return msg, gr.update(visible=True), df_data
        else:
             return msg, gr.update(visible=False), None

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
        
        # Always ensure the target dataset is selected in the backend
        app.select_dataset(target_dataset)
        
        # 3. Prepare Updates
        # Refresh datasets list in case a temp dataset was just created
        all_datasets = app.get_all_datasets()
        choices = [d["name"] for d in all_datasets]
        
        # Import helpers here to avoid circular dependencies or context issues
        from src.VLMAnnotate.ui_logic import load_selected_img, get_dataset_images_for_gallery
        
        img = load_selected_img(app) 
        gallery_imgs = get_dataset_images_for_gallery(app)

        vlm_comps = all_components["vlm_tab"]
        
        print(f"DEBUG on_annotate_click: target_dataset = {target_dataset}")
        print(f"DEBUG on_annotate_click: choices = {choices}")
        print(f"DEBUG on_annotate_click: app.selected_dataset = {app.selected_dataset}")
        
        return (
            gr.update(selected="vlm_tab"),   # Use explicit string ID  
            gr.update(choices=choices, value=target_dataset),
            img,    
            gallery_imgs
        )

    def upload_formatted_and_refresh(zip_file): 
        """Handle formatted dataset upload and refresh dropdowns"""
        success, message, dataset_name = app.save_formatted_dataset(zip_file)
        # Get updated list
        own_datasets = app.get_all_own_datasets()
        choices = [d["name"] for d in own_datasets]
        selected_value = dataset_name if success else None
        
        # Get path for selected dataset
        path_text = ""
        if selected_value:
            dataset = next((d for d in own_datasets if d["name"] == selected_value), None)
            if dataset:
                path_text = f"Path: {dataset['path']}"
        
        train_comps = all_components["train_tab"]
        
        return (
            message, 
            gr.update(choices=choices, value=selected_value),  # formatted_dataset_dropdown (Upload tab)
            gr.update(choices=choices, value=selected_value),  # formatted_dataset_dropdown_train (Train tab)
            path_text  # formatted_dataset_path_info
        )

    def delete_formatted_and_refresh(dataset_name): 
        """Handle formatted dataset deletion and refresh dropdowns"""
        success, message = app.delete_formatted_dataset(dataset_name)
        # Get updated list
        own_datasets = app.get_all_own_datasets()
        choices = [d["name"] for d in own_datasets]
        new_value = choices[0] if choices else None
        
        # Get path for new selected dataset
        path_text = ""
        if new_value:
            dataset = next((d for d in own_datasets if d["name"] == new_value), None)
            if dataset:
                path_text = f"Path: {dataset['path']}"
        
        train_comps = all_components["train_tab"]
        
        return (
            message, 
            gr.update(choices=choices, value=new_value),  # formatted_dataset_dropdown (Upload tab)
            gr.update(choices=choices, value=new_value),  # formatted_dataset_dropdown_train (Train tab)
            path_text  # formatted_dataset_path_info
        )

    def on_formatted_dataset_select(dataset_name): 
        """Update path info when formatted dataset is selected"""
        if not dataset_name:
            return ""
        own_datasets = app.get_all_own_datasets()
        dataset = next((d for d in own_datasets if d["name"] == dataset_name), None)
        if dataset:
            return f"Path: {dataset['path']}"
        return ""

    def update_dataset_dropdown(datasets):
        """Update dropdown choices"""
        choices = [d["name"] for d in datasets]
        # We need to return an update for the dropdown
        return gr.update(choices=choices, value=None)

    # --- Event Handlers ---
    c["dataset_dropdown"].change(
        fn=on_dataset_select,
        inputs=[c["dataset_dropdown"]],
        outputs=[c["status_output"], c["video_config_group"], c["video_dataframe"]]
    )

    vlm_comps = all_components["vlm_tab"]
    
    c["annotate_btn"].click(
        fn=on_annotate_click,
        inputs=[c["dataset_dropdown"], c["video_dataframe"], c["interval_slider"]],
        outputs=[
            all_components["tabs"],
            vlm_comps["vlm_dataset_dropdown"],
            vlm_comps["vlm_image_input"],
            vlm_comps["vlm_gallery"]
        ]
    )
    
    datasets_state = all_components["datasets_state"]

    c["delete_btn"].click(
        fn=lambda dataset_name, datasets: (
            delete_dataset_handler(dataset_name) if dataset_name 
            else ("Please select a dataset first", datasets)
        ),
        inputs=[c["dataset_dropdown"], datasets_state],
        outputs=[c["delete_status"], datasets_state]
    ).then(
        fn=update_dataset_dropdown,
        inputs=datasets_state,
        outputs=[c["dataset_dropdown"]]
    )
    
    c["upload_btn"].click(
        fn=upload_and_refresh,
        inputs=c["zip_file_input"],
        outputs=[c["upload_status"], datasets_state]
    ).then(
        fn=update_dataset_dropdown,
        inputs=datasets_state,
        outputs=[c["dataset_dropdown"]]
    )

    c["checkbox_formatted"].change(
        fn=toggle_button_mode,
        inputs=[c["checkbox_formatted"]],
        outputs=[c["dataset_format_dropdown"], c["upload_cvat_btn"], c["delete_btn"], c["annotate_btn"],c["dataset_dropdown"],c["status_output"],c["upload_btn"]]
    )
    
    c["upload_cvat_btn"].click(
        fn=upload_cvat_handler,
        inputs=[c["zip_file_input"], c["dataset_format_dropdown"]],
        outputs=[c["upload_cvat_status"]]
    )
    
    
