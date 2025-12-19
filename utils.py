import os
from pathlib import Path #handling file path 
from ultralytics import YOLO, settings #work with yolov8 model
from ultralytics.utils.checks import check_yolo, collect_system_info, cuda_device_count #verify YOLO environment is set up 
check_yolo(verbose=True, device='')
# print(collect_system_info())

import torch
if torch.cuda.is_available():
    print("✅ CUDA is available!")
    print(f"🖥️ GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"🔋 Memory Reserved : {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

class ProjectManager(): #manage all files and oflders related to a specific project (like 'semiconductor')
    def __init__(self, project_name):
        #create full absolute path for each directory
        self.train_dir = Path(r'/home/intern/Gitlab/pipeline/1.Train') / project_name
        self.eval_dir = Path(r'/home/intern/Gitlab/pipeline/2.Evaluation') / project_name
        self.dataset_dir = Path(r'/home/intern/Gitlab/pipeline/Dataset') / project_name
        
        #automatically creates these folders if they don't already exist, so you don't have to create them manually
        for dir in [self.train_dir, self.eval_dir, self.dataset_dir]:
            if not dir.exists():
                dir.mkdir(exist_ok=True, parents=True)
        (self.dataset_dir / 'images').mkdir(exist_ok=True, parents=True)

        #finds any pre-existing trained models (best.pt files) in your 1.Train folder.
        self.models = {Path(a).parent.parent.name: YOLO(a) 
                for a in self.train_dir.rglob('**/*best.pt')}
        
        #scans your Dataset folder to get a list of all your image files
        self.dataset = {d.name: [i for i in d.iterdir() if i.suffix in ['.jpg', '.png']]
                for d in (self.dataset_dir/'images').iterdir() if d.is_dir()}
        

    def check_project(self):
        print("Project Paths:")
        print(f"1. Training result: {self.train_dir}")
        print(f"2. Evaluation result: {self.eval_dir}")
        print(f"Dataset: {self.dataset_dir}")
        print('-----------------------------------------')
        print(f"Number of models: {len(self.models)}, {self.models.keys()}")
        print('-----------------------------------------')
        print("Dataset breakdown:")
        for key, data in self.dataset.items():
            print(f"{key}: {len(data)} Images")



def get_models(project):

    train_dir = Path(r'/home/intern/Gitlab/pipeline/1.Train') / project

    models = {Path(a).parent.parent.name: YOLO(a) 
              for a in train_dir.rglob('**/*best.pt')}
    return models

def get_dataset(project):

    dataset_dir = Path(r'/home/intern/Gitlab/pipeline/Dataset') / project

    image_dir = {d.name: [i for i in d.iterdir() if i.suffix in ['.jpg', '.png']]
        for d in (dataset_dir/'images').iterdir() if d.is_dir()}

    return image_dir

def get_train_dir(project):
    return Path(r'/home/intern/Gitlab/pipeline/1.Train') / project

def get_eval_dir(project):
    return Path(r'/home/intern/Gitlab/pipeline/2.Evaluation') / project

def get_data_dir(project):
    return Path(r'/home/intern/Gitlab/pipeline/Dataset') / project

if __name__ == '__main__':

    project = 'semiconductor7'
    print(Path(r'/home/intern/Gitlab/pipeline/Dataset') / project)
    print(Path(r'/home/intern/Gitlab/pipeline/1.Train') / project)
    print(Path(r'/home/intern/Gitlab/pipeline/2.Evaluation') / project)

    models = get_models(project)
    print(f"{len(models)} models: \n{models.keys()}")

    dataset = get_dataset(project)
    print(f"{len(dataset)} dataset: \n{dataset.keys()}")
    for key, data in dataset.items():
        print(f"{key}: {len(data)} Images")

