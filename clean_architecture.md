# Clean Architecture Documentation

This document outlines the reorganized, clean directory structure of the project.

## Directory Tree
`	ext
pipeline/
├── Dockerfile
├── LICENSE.txt
├── README.md
├── data
│   ├── database.db
│   ├── datasets
│   │   └── ... (dataset files)
│   ├── models
│   │   ├── pre_trained
│   │   │   └── ... (model files)
│   │   ├── train
│   │   │   └── ... (model files)
│   │   └── uploaded
│   │       └── ... (model files)
│   └── weights
│       └── ... (model files)
├── deploy_cpu.txt
├── deploy_cuda.txt
├── docs
│   └── assests
│       ├── architecture_diagram.drawio
│       └── system_context.svg
├── pyproject.toml
├── requirements-windows.txt
├── requirements.txt
└── src
    ├── apps
    │   ├── __init__.py
    │   └── video_to_dataset.py
    ├── seeai
    │   ├── __main__.py
    │   ├── config
    │   │   ├── __init__.py
    │   │   └── settings.py
    │   ├── core
    │   │   ├── __init__.py
    │   │   ├── annotation_engine.py
    │   │   ├── model_manager.py
    │   │   └── model_trainer.py
    │   ├── data
    │   │   ├── __init__.py
    │   │   ├── coco_builder.py
    │   │   ├── dataset_manager.py
    │   │   ├── db_manager.py
    │   │   └── yolo_builder.py
    │   ├── integrations
    │   │   ├── __init__.py
    │   │   └── cvat_client.py
    │   ├── models
    │   │   ├── __init__.py
    │   │   ├── grounding_dino.py
    │   │   └── qwen_vlm.py
    │   ├── ui
    │   │   ├── __init__.py
    │   │   ├── app.py
    │   │   ├── tabs
    │   │   │   ├── __init__.py
    │   │   │   ├── about_tab.py
    │   │   │   ├── generate_format_tab.py
    │   │   │   ├── predict_tab.py
    │   │   │   ├── pretrained_tab.py
    │   │   │   ├── train_tab.py
    │   │   │   └── vlm_tab.py
    │   │   └── ui_logic.py
    │   └── utils
    │       ├── __init__.py
    │       ├── file_utils.py
    │       ├── image_processing.py
    │       ├── qwen_processing.py
    │       ├── training_utils.py
    │       └── video_processing.py
    └── serving
        ├── __init__.py
        ├── server.py
        └── utils.py

`

## Structural Overview

1. **src/ (Source Code)**: The root for all Python application code.
   - **seeai/**: The main business logic and core features.
     - core/: Core classes like the annotation engine and model trainer.
     - data/: Dataset management and database handlers.
     - integrations/: Third-party services like the CVAT client.
     - models/: Wrappers for AI models (YOLO, Grounding DINO, Qwen).
     - ui/: Gradio user interface and tabs.
     - utils/: Helper scripts for video, image, and file processing.
     - config/: Application settings and .env.
   - **pps/**: Standalone applications, such as the ideo_to_dataset utility.
   - **serving/**: Server implementations and API endpoints.

2. **data/ (Application Data)**: Centralized storage for non-code assets (ignored by Git).
   - datasets/: Extracted datasets.
   - models/: Locally trained AI outputs (YOLO saves here automatically into train/).
   - weights/: Downloaded base model weights (sam2.1_b.pt, yolo11n.pt).
   - database.db: The SQLite registry for tracking model metadata.

3. **docs/ & examples/**: Project documentation, architecture diagrams, and tutorials.
