import os
from pathlib import Path
from ultralytics import YOLO, settings

import utils

def run_training(project_name, model_path, epochs, imgsz=640, manual_aug=False, aug_params=None):
    """
    Executes the training process.
    
    Args:
        project_name (str): Name of the project/dataset folder
        model_path (str): Path to the pre-trained model .pt file
        epochs (int): Number of epochs to train
        imgsz (int): Input image size
        manual_aug (bool): Whether to use manual augmentation override
        aug_params (dict): Dictionary of augmentation parameters
        model_output_name (str): Name for the trained model run
    """
    
    print(f"Starting training for project: {project_name}")
    print(f"Model: {model_path}, Epochs: {epochs}, Imgsz: {imgsz}, Manual Aug: {manual_aug}")

    # Initialize ProjectManager to get paths
    project = utils.ProjectManager(project_name)
    model_output_name = model_path.split("/")[-1].split(".")[0]
    
    # We no longer update global settings to avoid side-effects/stale configs
    # settings.update({ ... })
    
    try:
        # Load the model
        model = YOLO(model_path)
        
        # Prepare training arguments
        train_args = {
            'project': project.train_dir,           # Explicitly set output dir
            'data': project.dataset_dir / 'data.yaml', # Explicitly set data path
            'device': 0,
            'batch': -1,
            'patience': 300,
            'epochs': epochs,
            'workers': 1,
            'name': f'{model_output_name}_{epochs}',
            'save': True,
            'imgsz': imgsz,
        }
        
        # Apply manual augmentation overrides if enabled
        if manual_aug and aug_params:
            print("Applying manual augmentation parameters:", aug_params)
            train_args.update(aug_params)
        
        # Run training
        results = model.train(**train_args)
        
        print(f"Training completed. Results saved to {project.train_dir}")
        return True, f"Training completed successfully! Saved to {project.train_dir}"
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False, f"Training failed: {str(e)}"

if __name__ == '__main__':
    # Test run
    # Ensure you have a dataset named 'semiconductor7' set up if you run this directly
    run_training('semiconductor7', 'models/pre_trained/detection/yolov8n.pt', 1)   

    

