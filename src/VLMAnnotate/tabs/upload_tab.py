import gradio as gr
from src.VLMAnnotate.ui_logic import refresh_datasets_state
from src.VLMAnnotate.ui_logic import load_selected_img, get_dataset_images_for_gallery
def create_upload_components(app):
    """
    Creates the upload UI components without wrapping them in a Tab.
    This can be embedded into other tabs like vlm_tab.
    Returns a dictionary of components.
    """
    # ============================================================
    # 1. Upload Raw Dataset Section
    # ============================================================
    with gr.Row():
        # Left Column: Upload
        with gr.Column(scale=1):
            gr.Markdown("## Upload New Dataset")
            
            zip_file_input = gr.File(
                label="Upload Raw Dataset ZIP File", 
                file_count="single", 
                file_types=[".zip"], 
                type="filepath",
                height=207,
                visible=True
            )
            
            # State to track current zip/dataset
            # Stores: {"zip_path": str, "dataset_name": str, "has_videos": bool}
            current_dataset_state = gr.State(value={})

            
        with gr.Column():
            
            gr.Markdown("### ")
            checkbox_formatted = gr.Checkbox(
                label="Formatted Dataset", 
                interactive=True,
                value=False
            )
            click_instruction = gr.Markdown(
                    "### 👈 Please upload a dataset to begin",
                    visible=True
                )    
            # Formatted mode components
            dataset_format_dropdown = gr.Dropdown(
                choices=["COCO 1.0", "Ultralytics YOLO Classification 1.0", "Ultralytics YOLO Detection 1.0", "Ultralytics YOLO Segmentation 1.0"],
                value="COCO 1.0",
                label="Dataset Format",
                interactive=True,
                visible=False
            )
            
            upload_cvat_btn = gr.Button("📤 Upload to CVAT", variant="primary", size="lg", elem_id="btn", visible=False)
                
            upload_cvat_progress = gr.Textbox(
                label="Upload Progress",
                value="",
                interactive=False,
                visible=False,
                lines=1
            )
                
            upload_cvat_status = gr.Markdown(visible=False)

    # Video Frame Extraction Configuration
    with gr.Row():
        with gr.Group(visible=False) as video_config_group:
            gr.Markdown("### 🎥 Video Processing Detected")
            gr.Markdown("This dataset contains video files. Adjust extraction interval if needed.")
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
                label="Videos found in archive"
            )
            
            process_videos_btn = gr.Button("🎬 Process Videos", variant="primary", elem_id="btn")

    return {
        "zip_file_input": zip_file_input,
        "process_videos_btn": process_videos_btn,
        "video_config_group": video_config_group,
        "interval_slider": interval_slider,
        "video_dataframe": video_dataframe,
        "checkbox_formatted": checkbox_formatted,
        "dataset_format_dropdown": dataset_format_dropdown,
        "upload_cvat_btn": upload_cvat_btn,
        "upload_cvat_progress": upload_cvat_progress,
        "upload_cvat_status": upload_cvat_status,
        "current_dataset_state": current_dataset_state,
        "click_instruction": click_instruction
    }

def setup_upload_events(app, components, all_components):
    """
    Sets up event handlers for upload components.
    This can be called from other tabs that embed upload components.
    """
    # Unpack specific components needed
    c = components
    
    def toggle_button_mode(checked):
        if checked:
            return (
                gr.update(visible=True),   # dataset_format_dropdown
                gr.update(visible=True),   # upload_cvat_btn
                gr.update(label="Upload Formatted Dataset ZIP File"),  # zip_file_input (change label)
                gr.update(visible=False)   # click_instruction (hide for formatted)
            )
        else:
            return (
                gr.update(visible=False),  # dataset_format_dropdown
                gr.update(visible=False),  # upload_cvat_btn
                gr.update(label="Upload Raw Dataset ZIP File"),  # zip_file_input (change label)
                gr.update(visible=True)    # click_instruction
            )

    def upload_cvat_handler(zip_file, dataset_format):
        """Handle CVAT upload button click with progress updates"""
        if not zip_file:
            yield gr.update(visible=True, value="❌ Please upload a zip file first"), gr.update(visible=False)
            return
        
        # Show progress component - upload is starting
        yield gr.update(visible=True, value="📦 Uploading to CVAT... Please wait."), gr.update(visible=False)
        
        # Call the upload function (this will block until complete)
        # The progress parameter is still used for console logging
        success, message = app.upload_to_cvat(zip_file, dataset_format, progress=lambda p, desc="": None)
        
        # Hide progress, show final status
        yield gr.update(visible=False, value=""), gr.update(visible=True, value=message)

    # Internal logic functions
    def on_zip_changed(zip_file, state, is_formatted):
        """
        Triggered when a zip file is uploaded.
        - If is_formatted=False (raw upload): Process normally
        - If is_formatted=True (CVAT upload): Skip processing, just return waiting state
        """
        if not zip_file:
            return (
                "Waiting for upload...",
                gr.update(visible=False), # video_config
                gr.update(value=None),    # dataframe
                state,
                None,         # vlm_image_input
                []            # vlm_gallery
            )
        
        # If formatted dataset (for CVAT), skip raw dataset processing
        if is_formatted:
            return (
                "📦 Formatted dataset ready for CVAT upload",
                gr.update(visible=False),
                gr.update(value=None),
                state,
                None,
                []
            )
            
        # First, quick scan to check for videos before registering dataset
        import tempfile
        import zipfile
        from pathlib import Path
        
        temp_check_dir = Path(tempfile.mkdtemp(prefix="video_check_"))
        project_name = Path(zip_file).stem
        
        try:
            # Quick extract to check for videos
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_check_dir)
            
            # Scan for videos in temp location
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
            video_files_found = []
            for video_path in temp_check_dir.rglob('*'):
                if video_path.suffix.lower() in video_extensions:
                    video_files_found.append(video_path)
            
            has_videos = len(video_files_found) > 0
            
        finally:
            # Clean up temp check directory
            import shutil
            if temp_check_dir.exists():
                shutil.rmtree(temp_check_dir, ignore_errors=True)
        
        # Now register the dataset
        success, message, project_name = app.upload_dataset_by_zip(zip_file)
        
        if not success:
            return (
                f"❌ Error: {message}",
                gr.update(visible=False),
                gr.update(value=None),
                state,
                None,
                []
            )
            
        # Update state
        state = state or {}
        state["zip_path"] = zip_file
        state["dataset_name"] = project_name
        state["has_videos"] = has_videos
        
        if has_videos:
            # Has videos - show config, don't auto-select yet
            video_files = app.scan_for_videos(project_name)
            df_data = [[v["Video Name"], v["Duration"]] for v in video_files]
            
            return (
                f"✅ Ready: {project_name}\n(Videos detected - configure and process)",
                gr.update(visible=True),
                gr.update(value=df_data),
                state,
                None,
                []
            )
        else:
            # No videos - AUTO-SELECT dataset immediately
            app.select_dataset(project_name)
            
            # Load image and gallery for VLM tab
            img = load_selected_img(app)
            gallery_imgs = get_dataset_images_for_gallery(app)
            
            return (
                f"✅ Dataset selected: {project_name}",
                gr.update(visible=False),
                gr.update(value=None),
                state,
                img,
                gallery_imgs
            )
    def on_process_videos(state, video_df, interval_val):
        """
        Processes videos and auto-selects dataset.
        Triggered by Process Videos button.
        """
        if not state or not state.get("has_videos"):
            return (
                "No videos to process",
                None,
                []
            )
        
        dataset_name = state["dataset_name"]
        
        # Extract frames
        success, result = app.extract_frames_from_dataset(
            dataset_name, 
            video_df,
            interval_val
        )

        if success:
            temp_dataset_name = f"{dataset_name}_temp_frames"
            app.select_dataset(temp_dataset_name)

            # Clean up the original video dataset (no longer needed)
            app.remove_dataset(dataset_name)

            # Load image and gallery
            img = load_selected_img(app)
            gallery_imgs = get_dataset_images_for_gallery(app)

            return (
                f"✅ Dataset selected: {temp_dataset_name}",
                img,
                gallery_imgs
            )
        else:
            return (
                f"❌ Error processing videos: {result}",
                None,
                []
            )
    # --- Event Handlers ---
    
    # 1. Zip Upload Handler
    c["zip_file_input"].change(
        fn=on_zip_changed,
        inputs=[c["zip_file_input"], c["current_dataset_state"], c["checkbox_formatted"]],
        outputs=[
             c["click_instruction"],     # Status/Instruction
             c["video_config_group"],    # Visibility
             c["video_dataframe"],       # content
             c["current_dataset_state"], # State update
             all_components["vlm_tab"]["vlm_image_input"],   
             all_components["vlm_tab"]["vlm_gallery"]        
        ]
    )

    vlm_comps = all_components["vlm_tab"]

    c["process_videos_btn"].click(
        fn=on_process_videos,  
        inputs=[c["current_dataset_state"], c["video_dataframe"], c["interval_slider"]],
        outputs=[
            c["click_instruction"],
            all_components["vlm_tab"]["vlm_image_input"],
            all_components["vlm_tab"]["vlm_gallery"]
        ]
    )

    datasets_state = all_components["datasets_state"]

    c["checkbox_formatted"].change(
        fn=toggle_button_mode,
        inputs=[c["checkbox_formatted"]],
        outputs=[c["dataset_format_dropdown"], c["upload_cvat_btn"], c["zip_file_input"], c["click_instruction"]]
    )
    
    c["upload_cvat_btn"].click(
        fn=upload_cvat_handler,
        inputs=[c["zip_file_input"], c["dataset_format_dropdown"]],
        outputs=[c["upload_cvat_progress"], c["upload_cvat_status"]]
    )

def setup_events(app, components, all_components):
    """
    Wrapper function for backward compatibility.
    Calls setup_upload_events.
    """
    setup_upload_events(app, components, all_components)
