import gradio as gr

from . import utils

from .app import APP
from .page_datasets import load_dataset_interface
from .page_vlm_annotation import load_vlm_interface

def main():
    # Initialize the app
    model_id = "./huggingface/hub/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
    gdino = utils.GroundingDINODetector(model_id)
    app = APP(gdino)

    with gr.Blocks(theme=gr.themes.Soft(), title="See.AI Agent") as app_interface:
        load_dataset_interface(app_interface, app)

    with app_interface.route("vlm") as vlm_page:
        load_vlm_interface(vlm_page, app)

    app_interface.launch(
        debug=True,
        share=False,
        server_name="0.0.0.0",  # Allow external connections
        server_port=6605 #Testing Port
    )

if __name__ == '__main__':
    main()