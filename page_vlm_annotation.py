import gradio as gr
import os
from vlm_object_detection import GdinoModel
from PIL import Image

def load_vlm_interface(app_interface=None, app=None):

    gr.Markdown("# 🎯 See.AI Agent for Defect Inspection / Yield Improvement")
    gr.Markdown("## AI Agent to annotate data")
    gr.Markdown("Upload an image and describe what objects you want to detect!")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                value="./.gradio/VLM experiment.png",  # Put your sample image path here
                type="pil",
                label="Upload Image",
                height=400,
            )
            
            text_input = gr.Textbox(
                value="Black Resistor. Wires. Chips.",
                label="Text Prompt",
                placeholder="Describe objects to detect (e.g., 'a person. a car. a dog.')",
                lines=2
            )
            
            confidence_slider = gr.Slider(
                minimum=0.01,
                maximum=1.0,
                value=0.15,
                step=0.01,
                label="Confidence Threshold"
            )

            detect_btn = gr.Button("🔍 Detect Objects", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            sel_btn = gr.Button("Globe Dataset", variant="secondary")
            
            # Example prompts
            gr.Markdown("### 💡 Example Prompts:")
            example_prompts = [
                "Black Resistor. Wires. Chips.",
            ]
            
            for prompt in example_prompts:
                gr.Button(f"'{prompt}'", size="sm").click(
                    lambda p=prompt: p,
                    outputs=text_input
                )
        
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Detection Results",
                height=400
            )
            
            prompt_output = gr.Textbox(
                label="Parameters Used ",
                lines=2,
                show_copy_button=False
            )
            inference_output = gr.Textbox(
                label="Reference Output",
                lines=2,
                show_copy_button=False,
                value="Ready for inference on dataset"
            )
            inference_btn = gr.Button("Inference Dataset", variant="primary")
            download_btn = gr.DownloadButton("Download COCO Dataset", variant="primary", visible=False)

            detection_info = gr.Textbox(
                label="Detection Details",
                lines=10,
                show_copy_button=True
            )
            
            raw_output = gr.Textbox(
                label="Raw Results",
                lines=5,
                show_copy_button=True
            )


    # Event handlers
    detect_btn.click(
        fn=app.process_image,
        inputs=[image_input, text_input, confidence_slider],
        outputs=[output_image, detection_info, raw_output]
    ).then(
        fn=lambda prompt, confidence_slider: f"Prompt: {prompt}\nConfidence Threshold: {confidence_slider}",
        inputs=[text_input,confidence_slider],
        outputs=[prompt_output]
    )
    
    clear_btn.click(
        lambda: [None, "", "", None, "", ""],
        outputs=[image_input, text_input, confidence_slider, output_image, detection_info, raw_output]
    )

    sel_btn.click(
        fn=lambda: app.select_dataset('globe'),
        outputs=[inference_output]
    )

    inference_btn.click(
        fn=app.inference_dataset,
        inputs=[text_input, confidence_slider],
        outputs=[inference_output]
    ).then(
        fn=lambda: [gr.Button(visible=False), 
                    gr.DownloadButton(label=f"Download Dataset: {app.selected_dataset}", 
                                      value= f"./.output/{app.selected_dataset}_coco.zip", 
                                      visible=True)],
        outputs=[inference_btn, download_btn]
    )

    download_btn.click(
        fn=lambda: [gr.Button(visible=True), 
                    gr.DownloadButton(visible=False)],
        outputs=[inference_btn, download_btn]
    )
    
    # Allow Enter key to trigger detection
    text_input.submit(
        fn=app.process_image,
        inputs=[image_input, text_input, confidence_slider],
        outputs=[output_image, detection_info, raw_output]
    )
    
    # Instructions
    gr.Markdown("""
    ### 📝 How to Use:
    1. **Upload an image** using the image input
    2. **Enter a text prompt** describing what objects to detect
    3. **Adjust confidence threshold** (lower = more detections, higher = more confident detections)
    4. **Click "Detect Objects"** to run the detection
    
    ### 🎯 Prompt Tips:
    - Use simple object names separated by periods: "a person. a car. a dog."
    - Be specific: "a red car. a person walking. a bicycle."
    - You can detect multiple different objects in one prompt
    - The model works best with common objects and scenes
    """)
    def load_selected_img():

        if app.selected_dataset:
            dataset_name = app.selected_dataset
            dataset_img_path = app.selected_dataset_1st_img_path
            if os.path.exists(dataset_img_path) and os.path.splitext(dataset_img_path)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                return dataset_img_path
        else:
            return "./.gradio/VLM experiment.png"  # Default image path
        
    app_interface.load(
        fn=load_selected_img,
        outputs=image_input
    )
    

# Launch the app
if __name__ == "__main__":

    # Load model into device: GPU or CPU

    from app import APP
    from utils import GroundingDINODetector
    model_id = "./huggingface/hub/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
    gdino = GroundingDINODetector(model_id)
    app = APP(gdino)
    # app.select_dataset('gradio')

    # Create Gradio interface
    with gr.Blocks(title="See.AI Agent") as demo:
        load_vlm_interface(demo, app)

    demo.launch(
        debug=True,
        share=False,
        server_name="0.0.0.0",  # Allow external connections
        server_port=6605 #Testing Port
    )