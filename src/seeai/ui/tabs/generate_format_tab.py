import gradio as gr
from src.seeai.ui.ui_logic import load_selected_img, get_dataset_images_for_gallery
from pathlib import Path
import os


def create_tab(app):
    with gr.Tab("🎯 Generate Format Folder", id="generate_format_tab") as tab:
        # ============================================================
        # Upload Section (embedded from upload_tab)
        # ============================================================
        
        
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
        # QWEN Annotation Section
        # ============================================================
        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(scale=1):
                    gr.Markdown("### 🖼️ Dataset Gallery")
                    qwen_gallery = gr.Gallery(
                        label="Select Image from Dataset",
                        show_label=False,
                        columns=[4],
                        rows=[1],
                        height=150,
                        allow_preview=True,
                        interactive=True,
                    )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Sample Image")
                qwen_image_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    height=400,
                )
                    
                qwen_text_input = gr.Textbox(
                    value="person, car, traffic light, bicycle, stop sign",
                    label="Text Prompt (QWEN can handle long prompts)",
                    placeholder="Describe objects to detect (e.g., 'person, car, bicycle, traffic light')",
                    lines=3
                )

                gr.Markdown("""
                    > **Note**: QWEN doesn't use confidence thresholds. Detection quality depends on prompt clarity.
                """)

                detect_btn = gr.Button("🔍 Run On Image", variant="primary", size="lg", elem_id="btn")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", elem_id="del_btn", visible=False)
                    
                
            
            with gr.Column(scale=1):
                    
                gr.Markdown("### 📊 Output")
                prompt_output = gr.Textbox(
                    label="Prompt Used", 
                    lines=2, 
                    visible=False
                )
                qwen_output_image = gr.Image(label="Detection Results", height=400)
                
                inference_btn = gr.Button("⚡ Inference Full Dataset", variant="primary", elem_id="btn")
                
                inference_output = gr.HTML(
                    padding=False,
                    label="Inference Output",
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
                    height=150, 
                    visible=False
                )

                export_yolo_btn = gr.Button("📦 Export YOLO Format", visible=False, variant="primary")
                
                detection_info = gr.Textbox(label="Detection Details", lines=10, visible=False)
                raw_output = gr.Textbox(label="Raw QWEN Response", lines=5, visible=False)

    # Merge upload components with QWEN components
    components_dict = {
        "tab": tab,
        "qwen_gallery": qwen_gallery,
        "qwen_image_input": qwen_image_input,
        "qwen_text_input": qwen_text_input,
        "detect_btn": detect_btn,
        "clear_btn": clear_btn,
        "prompt_output": prompt_output,
        "qwen_output_image": qwen_output_image,
        "inference_btn": inference_btn,
        "inference_output": inference_output,
        "annotated_dir_path": annotated_dir_path,
        "annotated_gallery": annotated_gallery,
        "export_yolo_btn": export_yolo_btn,
        "detection_info": detection_info,
        "raw_output": raw_output
    }
    
    
    
    
    return components_dict


def setup_events(app, components, all_components):
    c = components
    
    
    
    def refresh_yolo_ui(inference_output):
        """Show YOLO export button only if inference succeeded"""
        # Check if output contains error (❌) or success
        if isinstance(inference_output, str) and "❌" in inference_output:
            # Inference failed - keep buttons as is
            return [
                gr.Button(visible=True),   # Keep inference button visible
                gr.Button(visible=False),  # Keep YOLO button hidden
                gr.Gallery(visible=False)
            ]
        else:
            # Inference succeeded - swap buttons
            return [
                gr.Button(visible=False),  # Hide inference button
                gr.Button(visible=True),   # Show YOLO button
                gr.Gallery(visible=True)
            ]

    def on_gallery_select(evt: gr.SelectData): 
        return evt.value["image"]["path"]

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

    # --- Event Handlers ---
    c["detect_btn"].click(
        fn=app.process_image_qwen,
        inputs=[c["qwen_image_input"], c["qwen_text_input"]],
        outputs=[c["qwen_output_image"], c["detection_info"], c["raw_output"]]
    ).then(
        fn=lambda prompt: f"Prompt: {prompt}",
        inputs=[c["qwen_text_input"]],
        outputs=[c["prompt_output"]]
    )

    c["clear_btn"].click(
        lambda: [None, "", None, "", ""],
        outputs=[c["qwen_image_input"], c["qwen_text_input"], c["qwen_output_image"], c["detection_info"], c["raw_output"]]
    )

    c["inference_btn"].click(
        fn=app.inference_dataset_qwen,
        inputs=[c["qwen_text_input"]],
        outputs=[c["inference_output"], c["annotated_dir_path"]]
    ).then(
        fn=load_annotated_images,
        inputs=[c["annotated_dir_path"]],
        outputs=[c["annotated_gallery"]]
    ).then(
        fn=refresh_yolo_ui,
        inputs=[c["inference_output"]],
        outputs=[c["inference_btn"], c["export_yolo_btn"], c["annotated_gallery"]]
    )

    c["export_yolo_btn"].click(
        fn=app.export_yolo_format,
        inputs=[],
        outputs=[c["inference_output"]]
    ).then(
        fn=lambda: [gr.Button(visible=True), gr.Button(visible=False)],
        outputs=[c["inference_btn"], c["export_yolo_btn"]]
    )
    
    c["qwen_text_input"].submit(
        fn=app.process_image_qwen,
        inputs=[c["qwen_image_input"], c["qwen_text_input"]],
        outputs=[c["qwen_output_image"], c["detection_info"], c["raw_output"]]
    )
    
    # Refresh image and gallery when tab is selected
    c["tab"].select(
        fn=lambda: load_selected_img(app), 
        outputs=[c["qwen_image_input"]]
    ).then(
        fn=lambda: get_dataset_images_for_gallery(app), 
        outputs=[c["qwen_gallery"]]
    )

    c["qwen_gallery"].select(
        fn=on_gallery_select,
        outputs=[c["qwen_image_input"]]
    )
