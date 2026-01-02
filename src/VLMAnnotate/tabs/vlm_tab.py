import gradio as gr
from src.VLMAnnotate.ui_logic import load_selected_img, get_dataset_images_for_gallery

def create_tab(app):
    with gr.Tab("📝 VLM Annotation", id="vlm_tab") as tab:
        # --- VLM Code Integration ---
        gr.Markdown("## AI Agent to annotate data")
        
        with gr.Row():
            with gr.Column(scale=1):
                # --- Dataset Info ---
                gr.Markdown("### 🗂️ Current Dataset")
                vlm_dataset_info = gr.Textbox(
                    label="Active Dataset",
                    value=getattr(app, 'current_dataset', None) or "None",
                    interactive=False
                )
             
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

                inference_format = gr.Dropdown(
                    choices=["Detection", "Segmentation", "Classification"],
                    value="Detection",
                    label="Inference Format",
                    interactive=True
                )

                # ---------------------------------
                    
                vlm_confidence_slider = gr.Slider(
                    minimum=0.01, maximum=1.0, value=0.15, step=0.01, label="Confidence Threshold"
                )

                detect_btn = gr.Button("🔍 Run On Image", variant="primary", size="lg", elem_id="btn")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", elem_id="del_btn", visible=False)
                    
                
            
            with gr.Column(scale=1):
                    
                gr.Markdown("### 📊 Output")
                prompt_output = gr.Textbox(
                    label="Parameters Used ", 
                    lines=2, 
                    show_copy_button=False,
                    visible=False
                )
                vlm_output_image = gr.Image(label="Detection Results", height=400)
                inference_btn = gr.Button("Inference Full Dataset", variant="primary", elem_id="btn")
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
                cvat_btn = gr.Button("Create CVAT Project and Task", visible=False, variant="primary")
                
                detection_info = gr.Textbox(label="Detection Details", lines=10, show_copy_button=True, visible=False)
                raw_output = gr.Textbox(label="Raw Results", lines=5, show_copy_button=True, visible=False)

    return {
        "tab": tab,
        "vlm_dataset_info": vlm_dataset_info,

        "vlm_gallery": vlm_gallery,
        "vlm_image_input": vlm_image_input,
        "vlm_text_input": vlm_text_input,
        "inference_format": inference_format,
        "vlm_confidence_slider": vlm_confidence_slider,
        "detect_btn": detect_btn,
        "clear_btn": clear_btn,
        "prompt_output": prompt_output,
        "vlm_output_image": vlm_output_image,
        "inference_btn": inference_btn,
        "inference_output": inference_output,
        "cvat_project_dropdown": cvat_project_dropdown,
        "cvat_btn": cvat_btn,
        "detection_info": detection_info,
        "raw_output": raw_output
    }

def setup_events(app, components, all_components):
    c = components
    
    # Internal logic
    def refresh_vlm_dropdown(): 
        """Refresh dataset list and return update"""
        all_datasets = app.get_all_datasets()
        choices = [d["name"] for d in all_datasets]
        # Keep currently selected value if valid, or just current app selection
        val = getattr(app, 'current_dataset', None)
        return gr.update(choices=choices, value=val)
    
    def refresh_cvat_dropdown():
        """Refresh CVAT projects dropdown after inference"""
        projects = app.get_cvat_projects()
        return [
            gr.Button(visible=False),  # Hide inference button
            gr.Button(visible=True),   # Show CVAT button
            gr.Dropdown(visible=True, choices=projects, interactive=True)  # Show and refresh dropdown
        ]


    def on_vlm_dataset_select(dataset_name): 
        if not dataset_name: return gr.update(visible=False), None
                
        # Always select the dataset so that subsequent components (gallery, image input) 
        # can access the correct path via app.selected_dataset
        app.select_dataset(dataset_name)

        video_files = app.scan_for_videos(dataset_name)
        has_videos = len(video_files) > 0
        if has_videos:
            df_data = [[v["Video Name"], v["Duration"]] for v in video_files]
            return gr.update(visible=True), df_data
        else:
            return gr.update(visible=False), None

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

    def on_gallery_select(evt: gr.SelectData): 
        return evt.value["image"]["path"]

    # --- Event Handlers ---
    c["detect_btn"].click(
        fn=app.process_image,
        inputs=[c["vlm_image_input"], c["vlm_text_input"], c["vlm_confidence_slider"], c["inference_format"]],
        outputs=[c["vlm_output_image"], c["detection_info"], c["raw_output"]]
    ).then(
        fn=lambda prompt, conf, fmt: f"Prompt: {prompt}\nConfidence: {conf}\nFormat: {fmt}",
        inputs=[c["vlm_text_input"], c["vlm_confidence_slider"], c["inference_format"]],
        outputs=[c["prompt_output"]]
    )

    c["clear_btn"].click(
        lambda: [None, "", 0.15, None, "", ""],
        outputs=[c["vlm_image_input"], c["vlm_text_input"], c["vlm_confidence_slider"], c["vlm_output_image"], c["detection_info"], c["raw_output"]]
    )

    c["inference_btn"].click(
        fn=app.inference_dataset,
        inputs=[c["vlm_text_input"], c["vlm_confidence_slider"], c["inference_format"]],
        outputs=[c["inference_output"]]
    ).then(
        fn=lambda _: refresh_cvat_dropdown(),
        outputs=[c["inference_btn"], c["cvat_btn"], c["cvat_project_dropdown"]]
    )

    c["cvat_btn"].click(
        fn=app.create_cvat_project_with_tasks,
        inputs=[c["cvat_project_dropdown"]],
        outputs=[c["inference_output"]]
    ).then(
        fn=lambda _: [gr.Button(visible=True), gr.Button(visible=False), gr.Dropdown(visible=False)],
        outputs=[c["inference_btn"], c["cvat_btn"], c["cvat_project_dropdown"]]
    )
    
    c["vlm_text_input"].submit(
        fn=app.process_image,
        inputs=[c["vlm_image_input"], c["vlm_text_input"], c["vlm_confidence_slider"], c["inference_format"]],
        outputs=[c["vlm_output_image"], c["detection_info"], c["raw_output"]]
    )
    
    # Refresh image and gallery when tab is selected (for cases where user navigates directly to VLM tab)
    c["tab"].select(
        fn=lambda: load_selected_img(app), 
        outputs=[c["vlm_image_input"]]
    ).then(
        fn=lambda: get_dataset_images_for_gallery(app), 
        outputs=[c["vlm_gallery"]]
    )

    c["vlm_gallery"].select(
        fn=on_gallery_select,
        outputs=[c["vlm_image_input"]]
    )
