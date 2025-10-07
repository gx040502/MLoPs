import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Union, Optional

class COCODatasetBuilder:
    """
    A class to build COCO format datasets incrementally.
    
    This class allows you to initialize a COCO dataset structure and 
    incrementally add images and annotations, then save to JSON.
    """
    
    def __init__(self, 
                 base_dir: Optional[Union[str, Path]] = 'coco_dataset',
                 contributor: str = "Chee Yee",
                 description: str = "Demo to construct COCO dataset",
                 version: str = "0.0.1",
                 categories: Optional[List[Dict]] = None,
                 create_subdirs: bool = True):
        """
        Initialize the COCO dataset builder.
        
        Args:
            contributor: Name of the dataset contributor
            description: Description of the dataset
            version: Version of the dataset
            categories: List of category dictionaries. If None, uses default Wire/Active regions.
        """
        # Use default categories if none provided
        if categories is None:
            categories = [
                {"id": 1, "name": "Wire Region", "supercategory": ""},
                {"id": 2, "name": "Active Region", "supercategory": ""}
            ]
        self.base_dir = Path(base_dir) if base_dir else None
        self.directories = {}

        if self.base_dir and create_subdirs:
            self._create_directory_structure()
        
        # Initialize COCO JSON structure
        self.coco_json = {
            "licenses": [{
                "name": "",
                "id": 0,
                "url": ""
            }],
            'info': {
                "contributor": contributor,
                "date_created": datetime.now().strftime("%d %b %Y"),
                "description": description,
                "url": "",
                "version": version,
                "year": str(datetime.now().year)
            },
            "categories": categories,
            "images": [],
            "annotations": []
        }
        
        # Track IDs for automatic increment
        self.next_image_id = 1
        self.next_annotation_id = 1
        
        # Keep track of used image IDs to prevent duplicates
        self.used_image_ids = set()
        
        # print(f"✅ COCO Dataset Builder initialized")
        # print(f"📋 Categories: {[cat['name'] for cat in categories]}")

    def _create_directory_structure(self):
        """Create the standard COCO dataset directory structure."""
        # Main directories
        self.directories = {
            'base': self.base_dir,
            'annotations': self.base_dir / 'annotations',
            'images': self.base_dir / 'images',
            'annotated_images': self.base_dir / 'images' / 'annotated',
            'train_images': self.base_dir / 'images' / 'Train',
            'val_images': self.base_dir / 'images' / 'Validation',
            'test_images': self.base_dir / 'images' / 'Test'
        }
        
        # Create all directories
        for dir_name, dir_path in self.directories.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            # print(f"📁 Created: {dir_path}")

    def get_directory_path(self, dir_type: str) -> Optional[Path]:
        """
        Get the path for a specific directory type.
        
        Args:
            dir_type: Type of directory ('base', 'annotations', 'images', 
                     'annotated_images', 'train_images', 'val_images', 'test_images')
        
        Returns:
            Path object or None if not found
        """
        return self.directories.get(dir_type)
    
    def add_custom_directory(self, dir_name: str, relative_path: str) -> Path:
        """
        Add a custom directory to the dataset structure.
        
        Args:
            dir_name: Name identifier for the directory
            relative_path: Path relative to base_dir
            
        Returns:
            Path object of the created directory
        """
        if not self.base_dir:
            raise ValueError("Base directory not set. Cannot create custom directories.")
        
        custom_dir = self.base_dir / relative_path
        custom_dir.mkdir(parents=True, exist_ok=True)
        self.directories[dir_name] = custom_dir
        
        print(f"📁 Added custom directory: {custom_dir}")
        return custom_dir
    
    def add_image(self, 
                  img_name: str,
                  img_width: int,
                  img_height: int,
                  verbose: bool = True) -> int:
        """
        Add an image to the COCO dataset.
        
        Args:
            img_name: Filename of the image
            img_width: Width of the image in pixels
            img_height: Height of the image in pixels
            img_id: Custom image ID (optional, auto-generated if None)
            
        Returns:
            The image ID that was used
        """
        img_id = self.next_image_id
        self.next_image_id += 1
        
        # Check for duplicate image IDs
        if img_id in self.used_image_ids:
            raise ValueError(f"Image ID {img_id} already exists in the dataset")
        
        self.used_image_ids.add(img_id)
        
        # Add image to COCO structure
        self.coco_json["images"].append({
            "id": img_id,
            "width": img_width,
            "height": img_height,
            "file_name": img_name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        })
        if verbose:
            print(f"📷 Added image: {img_name} (ID: {img_id}, Size: {img_width}x{img_height})")
        return img_id
    
    def add_annotation(self,
                      img_id: int,
                      category_id: int,
                      xywh: Union[List, Tuple],
                      segmentation: Optional[List] = None,
                      iscrowd: int = 0,
                      attributes: Optional[Dict] = None,
                      verbose: bool = True) -> int:
        """
        Add an annotation to the COCO dataset.
        
        Args:
            img_id: ID of the image this annotation belongs to
            category_id: ID of the category for this annotation
            xywh: Bounding box in [x, y, width, height] format
            annotation_id: Custom annotation ID (optional, auto-generated if None)
            segmentation: Segmentation data (optional, empty list if None)
            iscrowd: Whether this is a crowd annotation (0 or 1)
            attributes: Additional attributes dictionary
            
        Returns:
            The annotation ID that was used
        """
        # Validate image_id exists
        if img_id not in self.used_image_ids:
            raise ValueError(f"Image ID {img_id} not found in dataset. Add the image first.")
        
        # Validate category_id exists
        valid_category_ids = {cat["id"] for cat in self.coco_json["categories"]}
        if category_id not in valid_category_ids:
            raise ValueError(f"Category ID {category_id} not found. Valid IDs: {valid_category_ids}")
        
        # Use provided ID or auto-generate
        annotation_id = self.next_annotation_id
        self.next_annotation_id += 1
        
        # Convert xywh to list and calculate area
        x, y, w, h = xywh
        area = w * h
        
        # Default segmentation and attributes
        if segmentation is None:
            segmentation = []
        
        if attributes is None:
            attributes = {
                "occluded": "false",
                "rotation": 0.0
            }
        
        # Add annotation to COCO structure
        self.coco_json["annotations"].append({
            "id": annotation_id,
            "image_id": img_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "area": area,
            "bbox": [x, y, w, h],
            "iscrowd": iscrowd,
            "attributes": attributes
        })
        
        # Get category name for logging
        if verbose: 
            category_name = next(cat["name"] for cat in self.coco_json["categories"] if cat["id"] == category_id)
            print(f"🏷️  Added annotation: {category_name} (ID: {annotation_id}, bbox: [{x}, {y}, {w}, {h}])")
        
        return annotation_id
    
    def add_category(self, 
                    category_id: int,
                    name: str,
                    supercategory: str = "") -> None:
        """
        Add a new category to the dataset.
        
        Args:
            category_id: Unique ID for the category
            name: Name of the category
            supercategory: Parent category name
        """
        # Check if category ID already exists
        existing_ids = {cat["id"] for cat in self.coco_json["categories"]}
        if category_id in existing_ids:
            raise ValueError(f"Category ID {category_id} already exists")
        
        self.coco_json["categories"].append({
            "id": category_id,
            "name": name,
            "supercategory": supercategory
        })
        
        print(f"🏷️  Added category: {name} (ID: {category_id})")
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the current dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "total_images": len(self.coco_json["images"]),
            "total_annotations": len(self.coco_json["annotations"]),
            "total_categories": len(self.coco_json["categories"]),
            "categories": [cat["name"] for cat in self.coco_json["categories"]],
            "annotations_per_category": {},
            "next_image_id": self.next_image_id,
            "next_annotation_id": self.next_annotation_id
        }
        
        # Count annotations per category
        for annotation in self.coco_json["annotations"]:
            cat_id = annotation["category_id"]
            cat_name = next(cat["name"] for cat in self.coco_json["categories"] if cat["id"] == cat_id)
            stats["annotations_per_category"][cat_name] = stats["annotations_per_category"].get(cat_name, 0) + 1
        
        return stats
    
    def __call__(self) -> None:
        """Print dataset statistics in a formatted way."""
        stats = self.get_statistics()
        
        print("\n📊 COCO Dataset Statistics:")
        print(f"  📷 Images: {stats['total_images']}")
        print(f"  🏷️  Annotations: {stats['total_annotations']}")
        print(f"  📂 Categories: {stats['total_categories']}")
        print(f"  🔢 Next Image ID: {stats['next_image_id']}")
        print(f"  🔢 Next Annotation ID: {stats['next_annotation_id']}")
        
        if stats['annotations_per_category']:
            print("  📋 Annotations per category:")
            for cat_name, count in stats['annotations_per_category'].items():
                print(f"    - {cat_name}: {count}")
    
    def save_json(self, output_path: Union[str, Path]) -> None:
        """
        Save the COCO dataset to a JSON file.
        
        Args:
            output_path: Path where to save the JSON file
        """
        output_path = Path(output_path)
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.coco_json, f, indent=2, ensure_ascii=False)
        
        print(f"💾 COCO dataset saved to: {output_path}")
        self()
    
    def load_existing_json(self, json_path: Union[str, Path]) -> None:
        """
        Load an existing COCO JSON file to continue building on it.
        
        Args:
            json_path: Path to existing COCO JSON file
        """
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.coco_json = json.load(f)
        
        # Update tracking variables
        if self.coco_json["images"]:
            self.next_image_id = max(img["id"] for img in self.coco_json["images"]) + 1
            self.used_image_ids = {img["id"] for img in self.coco_json["images"]}
        
        if self.coco_json["annotations"]:
            self.next_annotation_id = max(ann["id"] for ann in self.coco_json["annotations"]) + 1
        
        print(f"📂 Loaded existing COCO dataset from: {json_path}")
    
    def reset_dataset(self) -> None:
        """Reset the dataset to empty state while keeping categories and info."""
        self.coco_json["images"] = []
        self.coco_json["annotations"] = []
        self.next_image_id = 1
        self.next_annotation_id = 1
        self.used_image_ids = set()
        
        print("🔄 Dataset reset - images and annotations cleared")
    
    def get_coco_json(self) -> Dict:
        """
        Get the current COCO JSON structure.
        
        Returns:
            Dictionary containing the COCO dataset
        """
        return self.coco_json.copy()


# Example usage
if __name__ == "__main__":
    # Initialize the builder
    builder = COCODatasetBuilder(
        contributor="Chee Yee",
        description="Wire and Active Region Detection Dataset",
        version="1.0.0",
        categories=[
                    {"id": 1, "name": "Wire Region", "supercategory": ""},
                    {"id": 2, "name": "Active Region", "supercategory": ""}
                ]
    )
    
    # Add some images
    img_id_1 = builder.add_image("image001.jpg", 1024, 768)
    img_id_2 = builder.add_image("image002.jpg", 800, 600)
    
    # Add annotations for the images
    builder.add_annotation(img_id_1, category_id=1, xywh=[100, 50, 200, 150])  # Wire Region
    builder.add_annotation(img_id_1, category_id=2, xywh=[300, 200, 100, 80])   # Active Region
    builder.add_annotation(img_id_2, category_id=1, xywh=[50, 100, 150, 120])   # Wire Region
    
    # Add a custom category
    builder.add_category(3, "Defect Region", "defect")
    builder.add_annotation(img_id_2, category_id=3, xywh=[400, 300, 80, 60])    # Defect Region
    
    # Print statistics
    builder()
    
    # Save to JSON
    builder.save_json("output/instances_Train.json")
    
    # Example of loading existing dataset and adding more data
    # builder2 = COCODatasetBuilder()
    # builder2.load_existing_json("output/dataset.json")
    # builder2.add_image("image003.jpg", 640, 480)
    # builder2.save_json("output/dataset_updated.json")