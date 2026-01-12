import gradio as gr

def create_tab(app):
    with gr.Tab("ℹ️ About") as tab:
        gr.Markdown("""
        ## 🚀 SeeAI Platform Guide
        
        Welcome to the SeeAI Platform! This comprehensive tool allows you to manage the entire computer vision lifecycle from dataset creation to model inference.

        ### ✨ Key Features
        1. **Multiple Dataset Types**: Support for both Raw images/videos and Pre-formatted datasets.
        2. **Intelligent Auto-Annotation**: built-in Grounding DINO integration to automatically label raw datasets.
        3. **CVAT Integration**: Seamless synchronization with CVAT for manual verification and correction of annotations.
        4. **Flexible Training**: Train YOLO models with either simplified presets or full manual configuration.
        5. **Instant Deployment**: Download trained `.pt` weights immediately after training.
        6. **Integrated Prediction**: Test your trained models directly within the app on images, videos, or archives.
        7. **Custom Model Support**: Upload and test external custom-trained YOLO models.
        8. **Broad Format Support**: Works with `.mp4`, `.webm`, `.zip` archives, and batch images.

        ---

        ### 👣 How to Use

        #### Phase 1: Data Preparation (Auto Annotation Tab)
        1. **Upload Data**: 
           - **Formatted**: Upload a standard YOLO dataset zip.
           - **Raw**: Upload raw images or video.
        2. **Process**:
           - If **Formatted**: The system validates and prepares it for CVAT.
           - If **Raw**: Use the **Auto Annotation** tools (Grounding DINO) to generate initial labels.
        3. **Verify in CVAT**: The dataset is automatically pushed to CVAT. Open CVAT to manually refine bounding boxes or classes.

        #### Phase 2: Model Training (Train Model Tab)
        4. **Select Project**: Choose the CVAT project you just verified.
        5. **Configure Training**: 
           - **Manual**: Adjust epochs, batch size, learning rate, and more.
           - **Auto**: Use default optimized settings.
        6. **Train**: Start the training process. You can monitor progress in real-time.
        7. **Download**: Once finished, download the trained `.pt` model file.

        #### Phase 3: Inference & Testing
        8. **Predict Tab**: Select your newly trained project and model to run tests on new images or videos.
        9. **Pre-Trained Models Tab**: Upload your own custom `.pt` files (e.g., trained elsewhere) to use the platform's inference interface.

        ---

        ### 📂 System File Structure
        
        The application is organized into the following structure:
        
        ```text
        AI_PROJECT
        ├── config                # Global configuration
        ├── data
        │   ├── datasets          # STORE: Formatted datasets synced with CVAT projects
        │   │   ├── temp          # TEMP: Used for unzipping uploads, video frames extraction, and intermediate processing
        │   │   └── .output       # HIDDEN UNTIL INFERENCING: Contains auto-annotation results (images & instances.json) from Grounding DINO
        │   └── models            
        │       ├── pre_trained   # STORE: Base YOLO models (v8/v11) used as starting points for training
        │       ├── train         # STORE: Your trained models, weights, and metrics (organized by Project)
        │       └── uploaded      # STORE: Custom external models you upload for inference
        └── src
            ├── all_utils         # Core utilities (CVAT, Video, Datasets)
            ├── ModelManager      # Backend logic for DB and Training
            └── SEEAI             # Frontend Application
                └── tabs          # UI Components for each tab
        ```
        """)
        
    return {"tab": tab}

def setup_events(app, components, all_components):
    c = components
    c["tab"].select(fn=lambda: app.cleanup_preview(), outputs=None)
