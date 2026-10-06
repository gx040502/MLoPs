import os
import shutil
import random
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image


class YOLODatasetBuilder:
    """
    A class to build YOLO format datasets incrementally.
    
    YOLO format structure:
    dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    ├── data.yaml
    └── classes.txt
    
    Label format (per line in .txt file):
    <class_id> <x_center> <y_center> <width> <height>
    All coordinates normalized to [0, 1]
    """
    
    def __init__(self,
                 base_dir: Union[str, Path],
                 class_names: List[str],
                 train_val_split: float = 0.8,
                 create_subdirs: bool = True):
        """
        Initialize the YOLO dataset builder.
        
        Args:
            base_dir: Base directory for the dataset
            class_names: List of class names
            train_val_split: Ratio for train/val split (default 0.8 = 80% train, 20% val)
            create_subdirs: Whether to create directory structure
        """
        self.base_dir = Path(base_dir)
        self.class_names = class_names
        self.train_val_split = train_val_split
        
        # Create class name to ID mapping
        self.class_to_id = {name: i for i, name in enumerate(class_names)}
        
        # Storage for images and annotations before split
        self.images = []  # List of (image_path, image_info)
        self.annotations = {}  # Dict: image_id -> list of annotations
        
        # Directory structure
        self.directories = {}
        
        if create_subdirs:
            self._create_directory_structure()
    
    def _create_directory_structure(self):
        """Create the standard YOLO dataset directory structure."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = {
            'base': self.base_dir,
            'images': self.base_dir / 'images',
            'labels': self.base_dir / 'labels',
            'train_images': self.base_dir / 'images' / 'train',
            'val_images': self.base_dir / 'images' / 'val',
            'train_labels': self.base_dir / 'labels' / 'train',
            'val_labels': self.base_dir / 'labels' / 'val',
        }
        
        for key, path in subdirs.items():
            path.mkdir(parents=True, exist_ok=True)
            self.directories[key] = path
        
        print(f"✅ Created YOLO dataset structure at: {self.base_dir}")
    
    def add_image(self, 
                  image_path: Union[str, Path],
                  image_id: Optional[int] = None) -> int:
        """
        Add an image to the dataset.
        
        Args:
            image_path: Path to the image file
            image_id: Optional image ID (auto-generated if None)
            
        Returns:
            The image ID that was used
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Get image dimensions
        img = Image.open(image_path)
        width, height = img.size
        
        # Auto-generate ID if not provided
        if image_id is None:
            image_id = len(self.images)
        
        # Store image info
        image_info = {
            'id': image_id,
            'file_name': image_path.name,
            'path': str(image_path),
            'width': width,
            'height': height
        }
        
        self.images.append((image_path, image_info))
        
        # Initialize empty annotation list for this image
        if image_id not in self.annotations:
            self.annotations[image_id] = []
        
        return image_id
    
    def add_annotation(self,
                      image_id: int,
                      class_name: str,
                      bbox_xyxy: List[int],
                      verbose: bool = False):
        """
        Add an annotation to an image.
        
        Args:
            image_id: ID of the image this annotation belongs to
            class_name: Name of the class
            bbox_xyxy: Bounding box in [x1, y1, x2, y2] format (absolute pixels)
            verbose: Whether to print verbose output
        """
        if image_id not in self.annotations:
            raise ValueError(f"Image ID {image_id} not found. Add image first.")
        
        if class_name not in self.class_to_id:
            raise ValueError(f"Class '{class_name}' not in class list: {self.class_names}")
        
        # Get image dimensions
        image_info = None
        for _, info in self.images:
            if info['id'] == image_id:
                image_info = info
                break
        
        if image_info is None:
            raise ValueError(f"Image ID {image_id} not found")
        
        img_width = image_info['width']
        img_height = image_info['height']
        
        # Convert xyxy to YOLO format (normalized xywh)
        x1, y1, x2, y2 = bbox_xyxy
        
        # Calculate center and dimensions
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # Normalize to [0, 1]
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height
        
        # Clamp to [0, 1]
        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))
        
        class_id = self.class_to_id[class_name]
        
        annotation = {
            'class_id': class_id,
            'class_name': class_name,
            'x_center': x_center_norm,
            'y_center': y_center_norm,
            'width': width_norm,
            'height': height_norm
        }
        
        self.annotations[image_id].append(annotation)
        
        if verbose:
            print(f"  Added annotation: {class_name} ({class_id}) - "
                  f"[{x_center_norm:.4f}, {y_center_norm:.4f}, {width_norm:.4f}, {height_norm:.4f}]")
    
    def split_and_save(self, seed: int = 42):
        """
        Split dataset into train/val and save to disk.
        
        Args:
            seed: Random seed for reproducibility
        """
        if not self.images:
            raise ValueError("No images added to dataset")
        
        print(f"\n📊 Splitting dataset (train: {self.train_val_split*100:.0f}%, "
              f"val: {(1-self.train_val_split)*100:.0f}%)...")
        
        # Shuffle images
        random.seed(seed)
        shuffled_images = self.images.copy()
        random.shuffle(shuffled_images)
        
        # Split
        split_idx = int(len(shuffled_images) * self.train_val_split)
        train_images = shuffled_images[:split_idx]
        val_images = shuffled_images[split_idx:]
        
        print(f"  Train: {len(train_images)} images")
        print(f"  Val: {len(val_images)} images")
        
        # Save train set
        self._save_split(train_images, 'train')
        
        # Save val set
        self._save_split(val_images, 'val')
        
        # Save data.yaml
        self._save_data_yaml()
        
        # Save classes.txt
        self._save_classes_txt()
        
        print(f"\n✅ Dataset saved to: {self.base_dir}")
    
    def _save_split(self, images: List[Tuple[Path, Dict]], split: str):
        """Save a split (train or val) to disk."""
        images_dir = self.directories[f'{split}_images']
        labels_dir = self.directories[f'{split}_labels']
        
        for image_path, image_info in images:
            image_id = image_info['id']
            
            # Copy image
            dest_image = images_dir / image_info['file_name']
            shutil.copy2(image_path, dest_image)
            
            # Write label file
            label_file = labels_dir / (image_path.stem + '.txt')
            
            annotations = self.annotations.get(image_id, [])
            
            with open(label_file, 'w') as f:
                for ann in annotations:
                    # YOLO format: class_id x_center y_center width height
                    line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} " \
                           f"{ann['width']:.6f} {ann['height']:.6f}\n"
                    f.write(line)
    
    def _save_data_yaml(self):
        """Save data.yaml configuration file."""
        data_yaml = {
            'path': str(self.base_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': {i: name for i, name in enumerate(self.class_names)}
        }
        
        yaml_path = self.base_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
        
        print(f"  ✅ Saved data.yaml")
    
    def _save_classes_txt(self):
        """Save classes.txt file."""
        classes_path = self.base_dir / 'classes.txt'
        with open(classes_path, 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        
        print(f"  ✅ Saved classes.txt")
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the current dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        total_annotations = sum(len(anns) for anns in self.annotations.values())
        
        # Count annotations per class
        class_counts = {name: 0 for name in self.class_names}
        for anns in self.annotations.values():
            for ann in anns:
                class_counts[ann['class_name']] += 1
        
        return {
            'total_images': len(self.images),
            'total_annotations': total_annotations,
            'class_counts': class_counts,
            'num_classes': len(self.class_names)
        }
    
    def __call__(self):
        """Print dataset statistics in a formatted way."""
        stats = self.get_statistics()
        
        print("\n" + "="*50)
        print("YOLO Dataset Statistics")
        print("="*50)
        print(f"Total Images: {stats['total_images']}")
        print(f"Total Annotations: {stats['total_annotations']}")
        print(f"Number of Classes: {stats['num_classes']}")
        print("\nAnnotations per Class:")
        for class_name, count in stats['class_counts'].items():
            print(f"  {class_name}: {count}")
        print("="*50 + "\n")


if __name__ == "__main__":
    """Example usage of the YOLODatasetBuilder class."""
    
    # Initialize builder
    builder = YOLODatasetBuilder(
        base_dir="test_yolo_dataset",
        class_names=['person', 'car', 'bicycle'],
        train_val_split=0.8
    )
    
    # Add images and annotations (example)
    # img_id = builder.add_image("path/to/image1.jpg")
    # builder.add_annotation(img_id, 'person', [100, 200, 150, 300])
    # builder.add_annotation(img_id, 'car', [300, 400, 450, 550])
    
    # Split and save
    # builder.split_and_save()
    
    # Print statistics
    # builder()
    
    print("YOLODatasetBuilder initialized successfully!")
