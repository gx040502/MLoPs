import os
from pathlib import Path
from ultralytics import YOLO, settings
import time
import json

def run_training(project_name, model_path, epochs, imgsz=640, manual_aug=False, aug_params=None, format_name="Ultralytics YOLO Detection 1.0"):
    """
    Executes the training process.
    
    Args:
        project_name (str): Name of the project/dataset folder
        model_path (str): Path to the pre-trained model .pt file
        epochs (int): Number of epochs to train
        imgsz (int): Input image size
        manual_aug (bool): Whether to use manual augmentation override
        aug_params (dict): Dictionary of augmentation parameters
        format_name (str): Format of the dataset (Detection/Segmentation/Classification)
        model_output_name (str): Name for the trained model run
    """
    
    print(f"Starting training for project: {project_name}")
    print(f"Model: {model_path}, Epochs: {epochs}, Imgsz: {imgsz}, Manual Aug: {manual_aug}, Format: {format_name}")

    base_dir = Path(__file__).resolve().parents[2].joinpath('data')
    dataset_dir = base_dir / 'datasets' / project_name
    output_dir = base_dir / 'models/train' / project_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_output_name = Path(model_path).stem  # Works on all OS
    
    # We no longer update global settings to avoid side-effects/stale configs
    # settings.update({ ... })
    
    try:
        # Load the model
        model = YOLO(model_path)

        # --- ETA CALLBACK DEFINITION ---
        class TrainingCallback:
            def __init__(self, total_epochs, project_dir):
                self.total_epochs = total_epochs
                self.project_dir = Path(project_dir)
                self.epoch_start_time = 0
                self.first_epoch_duration = None
                
            def on_train_epoch_start(self, trainer):
                self.epoch_start_time = time.time()
                
            def on_train_epoch_end(self, trainer):
                duration = time.time() - self.epoch_start_time
                current_epoch = trainer.epoch + 1 # 1-indexed for display
                
                # Logic: Use FIRST epoch duration as baseline for everything
                if self.first_epoch_duration is None:
                    self.first_epoch_duration = duration
                
                # Estimated TOTAL time
                est_total_time = self.first_epoch_duration * self.total_epochs
                
                # Estimated REMAINING time
                # We subtract the time already spent (which is approx first_epoch * current_epoch for this simple logic
                # OR we just do first_epoch * remaining_epochs)
                remaining_epochs = self.total_epochs - current_epoch
                est_remaining_time = self.first_epoch_duration * remaining_epochs
                
                status_data = {
                    "current_epoch": current_epoch,
                    "total_epochs": self.total_epochs,
                    "last_epoch_duration": duration,
                    "first_epoch_duration": self.first_epoch_duration,
                    "est_total_time": est_total_time,
                    "est_time_remaining": est_remaining_time
                }
                
                # Write to file
                status_file = self.project_dir / "training_status.json"
                try:
                    with open(status_file, 'w') as f:
                        json.dump(status_data, f)
                except Exception as e:
                    print(f"Warning: Could not write training status: {e}")

        # Instantiate callback
        # We need the project dir to save the status file. 
        # project.train_dir is where results go (e.g. Dataset/Task_706/runs/detect/train)
        # We want the status file to be easily accessible. Let's put it in the project root (Dataset/Task_706)
        # project.dataset_dir is Dataset/Task_706
        
        callback = TrainingCallback(epochs, dataset_dir.as_posix())
        
        # Register callbacks
        model.add_callback("on_train_epoch_start", callback.on_train_epoch_start)
        model.add_callback("on_train_epoch_end", callback.on_train_epoch_end)

        
        # Determine data path based on format
        is_classification = "Classification" in format_name
        data_path = dataset_dir if is_classification else dataset_dir / 'data.yaml'
        
        # Prepare training arguments
        train_args = {
            'project': str(output_dir),           # Explicitly set output dir (Windows compatible)
            'data': data_path,                      # Directory for classification, yaml for others
            'device': 0,
            'batch': -1,
            'patience': epochs,
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

        print(f"Training completed. Results saved to {output_dir}")
        return True, f"Training completed successfully! Saved to {output_dir}"
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False, f"Training failed: {str(e)}"

if __name__ == '__main__':
    # Test run
    # Ensure you have a dataset named 'pipeline-dataset_107' set up if you run this directly
    run_training('pipeline-dataset_107', 'models/pre_trained/detection/yolov8n.pt', 1)   

    

