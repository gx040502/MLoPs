import gradio as gr
import os
import json
import pandas as pd

def refresh_datasets_state(app): 
    """Get current datasets and return as state"""
    return app.get_all_datasets()

def refresh_all_components(app, cvat_projects_dropdown_value): 
    """Refresh all dataset-related components"""
    updated_datasets = refresh_datasets_state(app)
    choices = [d["name"] for d in updated_datasets]
    first_dataset = choices[0] if choices else None
    
    # Select the first dataset in the backend
    if first_dataset:
        app.select_dataset(first_dataset)
    
    dropdown_update = gr.update(choices=choices, value=first_dataset)
    
    return updated_datasets

def load_selected_img(app): 
    if app.selected_dataset:
        dataset_img_path = app.selected_dataset_1st_img_path
        if os.path.exists(dataset_img_path) and os.path.splitext(dataset_img_path)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
            return dataset_img_path
    return None 

def get_dataset_images_for_gallery(app): 
    if not app.selected_dataset:
        return []
    success, dataset = app.get_dataset_by_name(app.selected_dataset)
    if not success: return []
    
    path = dataset.get("path")
    if not path or not os.path.exists(path): return []
            
    images = []
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    for f in sorted(os.listdir(path)):
        if f.lower().endswith(valid_ext):
            images.append(os.path.join(path, f))
    return images
