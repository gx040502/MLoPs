import gradio as gr
from src.seeai.ui.ui_logic import load_selected_img, get_dataset_images_for_gallery
from pathlib import Path
import os
def create_tab(app):
    with gr.Tab("✨ Auto Annotation", id="vlm_tab") as tab:
        # ============================================================
        # Upload Section (embedded from upload_tab)
        # ============================================================
        upload_components = create_upload_components(app)
        
        # Add separator
        gr.HTML("""
            <div style="text-align: center; margin: 20px 0;">
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
        
        # ============================================================
        # VLM Annotation Section
        # ============================================================
        
        with gr.Row():
            with gr.Column(scale=1):
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
                
                # Hidden textbox to store the annotated images directory path
                annotated_dir_path = gr.Textbox(visible=False)
                
                # Gallery to display all annotated images
                annotated_gallery = gr.Gallery(
                    label="Annotated Images",
                    show_label=True,
                    columns=4,
                    height= 150, 
                    visible=False
                )


                cvat_btn = gr.Button("Create CVAT Project and Task", visible=False, variant="primary")
                
                detection_info = gr.Textbox(label="Detection Details", lines=10, visible=False)
                raw_output = gr.Textbox(label="Raw Results", lines=5, visible=False)

    # Merge upload components with VLM components
    components_dict = {
        "tab": tab,
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
        "annotated_dir_path": annotated_dir_path,
        "annotated_gallery": annotated_gallery,
        "cvat_btn": cvat_btn,
        "detection_info": detection_info,
        "raw_output": raw_output
    }
    
    # Add upload components
    components_dict.update(upload_components)
    
    return components_dict

def setup_events(app, components, all_components):
    c = components
    
    # Set up upload event handlers
    setup_upload_events(app, components, all_components)
    
    def refresh_cvat_ui(inference_output):
        """Show CVAT button only if inference succeeded"""
        # Check if output contains error (❌) or success
        if isinstance(inference_output, str) and "❌" in inference_output:
            # Inference failed - keep buttons as is
            return [
                gr.Button(visible=True),   # Keep inference button visible
                gr.Button(visible=False),  # Keep CVAT button hidden
                gr.Gallery(visible=False)
            ]
        else:
            # Inference succeeded - swap buttons
            return [
                gr.Button(visible=False),  # Hide inference button
                gr.Button(visible=True),   # Show CVAT button
                gr.Gallery(visible=True)
            ]

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

    def load_annotated_images(annotated_dir):
        """Load all images from the annotated images directory for gallery display"""
        if not annotated_dir or annotated_dir == "":
            return []  # Return empty list for gallery
        
        dir_path = Path(annotated_dir)
        print(dir_path)
        
        if not dir_path.exists():
            return []  # Return empty list
        
        images = []
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        for f in sorted(os.listdir(dir_path)):
            if f.lower().endswith(valid_ext):
                images.append(os.path.join(dir_path, f))
        return images

    c["inference_btn"].click(
        fn=app.inference_dataset,
        inputs=[c["vlm_text_input"], c["vlm_confidence_slider"], c["inference_format"]],
        outputs=[c["inference_output"], c["annotated_dir_path"]]  # Store dir path in hidden textbox
    ).then(
        fn=load_annotated_images,
        inputs=[c["annotated_dir_path"]],  # Read dir path from hidden textbox
        outputs=[c["annotated_gallery"]]    # Update gallery with image list
    ).then(
        fn=refresh_cvat_ui,
        inputs=[c["inference_output"]],
        outputs=[c["inference_btn"], c["cvat_btn"], c["annotated_gallery"]]
    )

    c["cvat_btn"].click(
        fn=app.create_cvat_project_with_tasks,
        inputs=[],
        outputs=[c["inference_output"]]
    ).then(
        fn=lambda: [gr.Button(visible=True), gr.Button(visible=False)],
        outputs=[c["inference_btn"], c["cvat_btn"]]
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
