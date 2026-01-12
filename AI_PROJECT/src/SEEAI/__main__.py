import gradio as gr
import os
import threading
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from AI_PROJECT.src.all_utils import model as utils
from AI_PROJECT.src.all_utils.app import APP
from .page_datasets import load_dataset_interface

os.environ["GRADIO_TEMP_DIR"] = ".gradio_tmp/"
os.makedirs(".gradio_tmp/", exist_ok=True)

def cleanup_gradio_tmp():
    """Clean up old files in .gradio_tmp directory"""
    temp_dir = Path(".gradio_tmp")
    if temp_dir.exists():
        try:
            # Remove all contents
            for item in temp_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            print(f"✅ Cleaned up .gradio_tmp directory")
        except Exception as e:
            print(f"⚠️ Error cleaning .gradio_tmp: {e}")

def start_cleanup_timer():
    """Run cleanup once daily at 12 AM (midnight) in background"""
    while True:
        now = datetime.now()
        # Calculate next midnight
        next_run = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Calculate delay in seconds
        delay = (next_run - now).total_seconds()
        
        print(f"🧹 Cleanup scheduled in {delay/3600:.2f} hours (at midnight)")
        time.sleep(delay)
        
        cleanup_gradio_tmp()
 
def main():
    # Start background cleanup thread
    cleanup_thread = threading.Thread(target=start_cleanup_timer, daemon=True)
    cleanup_thread.start()
    print("🧹 Started automatic cleanup of .gradio_tmp (Daily at 12 AM)")
    
    # Initialize the app
    model_id = "IDEA-Research/grounding-dino-base"
    gdino = utils.GroundingDINODetector(model_id)
    app = APP(gdino)

    custom_css = """
    #detection_details_code .cm-scroller {
        max-height: 180px !important; 
        overflow-y: auto !important;
    }
    #cvat_project_dd .wrap,
    #cvat_project_dd .wrap input {
         min-height: 26px !important; 
    }
    """

    with gr.Blocks(title="See.AI Agent", fill_height=True) as app_interface:
        load_dataset_interface(app_interface, app)


    # Get absolute path to .gradio folder to satisfy allowed_paths requirements
    # Assuming CWD is the project root /home/intern/Gitlab/pipeline
    gradio_dir = os.path.join(os.getcwd(), ".gradio")
    
    
    app_interface.launch(
        debug=True,
        share=True,
        server_name="0.0.0.0",  # Allow external connections
        server_port=6605, #Testing Port
        allowed_paths=[gradio_dir],
        css= custom_css,
        theme = gr.themes.Soft()
    )
#testing
if __name__ == '__main__':
    main()
