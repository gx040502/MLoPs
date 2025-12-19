import gradio as gr
import os
from . import utils

from .app import APP
from .page_datasets import load_dataset_interface
from .page_vlm_annotation import load_vlm_interface

def main():
    # Initialize the app
    model_id = "IDEA-Research/grounding-dino-base"
    gdino = utils.GroundingDINODetector(model_id)
    app = APP(gdino)

    with gr.Blocks(theme=gr.themes.Soft(), title="See.AI Agent") as app_interface:
        load_dataset_interface(app_interface, app)

    with app_interface.route("vlm") as vlm_page:
        load_vlm_interface(vlm_page, app)

    # Get absolute path to .gradio folder to satisfy allowed_paths requirements
    # Assuming CWD is the project root /home/intern/Gitlab/pipeline
    gradio_dir = os.path.join(os.getcwd(), ".gradio")
    
    
    app_interface.launch(
        debug=True,
        share=False,
        server_name="0.0.0.0",  # Allow external connections
        server_port=6605, #Testing Port
        allowed_paths=[gradio_dir]
    )

if __name__ == '__main__':
    main()