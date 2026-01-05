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
                    label="Upload Raw Dataset ZIP File", 
                    file_count="single", 
                    file_types=[".zip"], 
                    type="filepath",
                    height=207,
                    visible=True
                )

                zip_file_formatted = gr.File(
                    label="Upload Formatted Dataset ZIP File", 
                    file_count="single", 
                    file_types=[".zip"], 
                    type="filepath",
                    height=207,
                    visible=False
                )

            
                gr.Markdown("## Status")
                
                # State to track current zip/dataset
                # Stores: {"zip_path": str, "dataset_name": str, "has_videos": bool}
                current_dataset_state = gr.State(value={})

                click_instruction = gr.Markdown(
                     "### 👈 Please upload a dataset to begin",
                     visible=True
                )

                checkbox_formatted = gr.Checkbox(
                    label="Formatted Dataset", 
                    interactive=True,
                    value=False
                )
                
                # Formatted mode components
                dataset_format_dropdown = gr.Dropdown(
                    choices=["COCO 1.0", "Ultralytics YOLO Classification 1.0", "Ultralytics YOLO Detection 1.0", "Ultralytics YOLO Segmentation 1.0"],
                    value="COCO 1.0",
                    label="Dataset Format",
                    interactive=True,
                    visible=False
                )
                
                with gr.Row():
                    annotate_btn = gr.Button("📂 Annotate", variant="primary", elem_id="btn", visible=True, interactive=False)
                    upload_cvat_btn = gr.Button("📤 Upload to CVAT", variant="primary", size="lg", elem_id="btn",visible=False)
                
                upload_cvat_progress = gr.Textbox(
                    label="Upload Progress",
                    value="",
                    interactive=False,
                    visible=False,
                    lines=1
                )
                
                upload_cvat_status = gr.Markdown(visible=False)

            # Video Frame Extraction Configuration
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

    return {
        "tab": tab,
        "zip_file_input": zip_file_input,
        "zip_file_formatted": zip_file_formatted,
        "annotate_btn": annotate_btn,
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

def setup_events(app, components, all_components):
    # Unpack specific components needed for this tab
    c = components
    
    def toggle_button_mode(checked):
        if checked:
            return (
                gr.update(visible=True),   # dataset_format_dropdown
                gr.update(visible=True),   # upload_cvat_btn
                gr.update(visible=False),  # annotate_btn
                gr.update(visible=True),   # zip_file_formatted
                gr.update(visible=False),  # zip_file_input
                gr.update(visible=False),   # click_instruction (hide for formatted)
                gr.update(visible=False)    # video_config_group (hide)
            )
        else:
            return (
                gr.update(visible=False),  # dataset_format_dropdown
                gr.update(visible=False),  # upload_cvat_btn
                gr.update(visible=True),   # annotate_btn
                gr.update(visible=False),  # zip_file_formatted
                gr.update(visible=True),   # zip_file_input
                gr.update(visible=True),    # click_instruction
                gr.update(visible=False)    # video_config_group (hide initially)
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
    def on_zip_changed(zip_file, state):
        """
        Triggered when a raw zip file is uploaded.
        1. Uploads dataset (flattening logic included).
        2. Scans for videos.
        3. Updates State and UI.
        """
        if not zip_file:
            return (
                "Waiting for upload...",
                gr.update(visible=False), # video_config
                gr.update(value=None),    # dataframe
                gr.update(interactive=False), # annotate_btn
                state
            )
            
        success, message, project_name = app.upload_dataset_by_zip(zip_file)
        
        if not success:
            return (
                f"❌ Error: {message}",
                gr.update(visible=False),
                gr.update(value=None),
                gr.update(interactive=False),
                state
            )
            
        # Scan for videos
        state = state or {}
        state["zip_path"] = zip_file
        state["dataset_name"] = project_name
        
        video_files = app.scan_for_videos(project_name)
        has_videos = len(video_files) > 0
        state["has_videos"] = has_videos
        
        df_data = None
        video_visible = False
        
        if has_videos:
            df_data = [[v["Video Name"], v["Duration"]] for v in video_files]
            video_visible = True
            
        return (
            f"✅ Ready: {project_name}\n" + ("(Videos detected)" if has_videos else ""),
            gr.update(visible=video_visible),
            gr.update(value=df_data),
            gr.update(interactive=True),
            state
        )

    def on_annotate_click(state, video_df, interval_val):
        """
        Handles annotation logic:
        1. If videos -> Extract frames -> New Zip -> Replace -> Select
        2. If no videos -> Use original Zip -> Select
        3. Switch Tab
        """
        if not state or not state.get("dataset_name"):
            return (
                gr.update(), # No tab switch
                gr.update(), # No dropdown
                gr.update(), # No image
                gr.update()  # No gallery
            )
            
        dataset_name = state["dataset_name"]
        has_videos = state.get("has_videos", False)
        
        final_dataset_name = dataset_name
        
        if has_videos:
            # Need to process videos
             # This creates a NEW temp dataset (and returns the zip path) AND selects it
             success, result = app.extract_frames_from_dataset(dataset_name, video_df, interval_val=interval_val)
             if success:
                 # Result is the path to the new zip file.
                 # The extract_frames_from_dataset ALREADY updated the config and selected it.
                 # The 'result' is the zip_path. The function extract_frames_from_dataset 
                 # also calls select_dataset(temp_name) internally.
                 # We just need to know the name. 
                 # Wait, extract_frames_from_dataset selects the dataset by name.
                 # Let's verify what it returns. It returns (True, zip_path).
                 # And it selects the dataset named f"{dataset_name}_temp_frames"
                 final_dataset_name = f"{dataset_name}_temp_frames"
        else:
             # Just ensure selected
             app.select_dataset(final_dataset_name)
        
        # 3. Prepare Updates
        all_datasets = app.get_all_datasets()
        choices = [d["name"] for d in all_datasets]
        
        from src.VLMAnnotate.ui_logic import load_selected_img, get_dataset_images_for_gallery
        
        img = load_selected_img(app) 
        gallery_imgs = get_dataset_images_for_gallery(app)
        
        vlm_comps = all_components["vlm_tab"]
        
        return (
            gr.update(selected="vlm_tab"),   
            gr.update(value=final_dataset_name),  # vlm_dataset_info is a Textbox, not Dropdown
            img,    
            gallery_imgs
        )

    # --- Event Handlers ---
    
    # 1. Zip Upload Handler
    c["zip_file_input"].change(
        fn=on_zip_changed,
        inputs=[c["zip_file_input"], c["current_dataset_state"]],
        outputs=[
             c["click_instruction"],     # Status/Instruction
             c["video_config_group"],    # Visibility
             c["video_dataframe"],       # content
             c["annotate_btn"],          # interactivity
             c["current_dataset_state"]  # State update
        ]
    )

    vlm_comps = all_components["vlm_tab"]
    
    # 2. Annotate Click Handler
    c["annotate_btn"].click(
        fn=on_annotate_click,
        inputs=[c["current_dataset_state"], c["video_dataframe"], c["interval_slider"]],
        outputs=[
            all_components["tabs"],
            vlm_comps["vlm_dataset_info"],
            vlm_comps["vlm_image_input"],
            vlm_comps["vlm_gallery"]
        ]
    )

    datasets_state = all_components["datasets_state"]

    c["checkbox_formatted"].change(
        fn=toggle_button_mode,
        inputs=[c["checkbox_formatted"]],
        outputs=[c["dataset_format_dropdown"], c["upload_cvat_btn"], c["annotate_btn"],c["zip_file_formatted"],c["zip_file_input"], c["click_instruction"], c["video_config_group"]]
    )
    
    c["upload_cvat_btn"].click(
        fn=upload_cvat_handler,
        inputs=[c["zip_file_formatted"], c["dataset_format_dropdown"]],
        outputs=[c["upload_cvat_progress"], c["upload_cvat_status"]]
    )
    
    
