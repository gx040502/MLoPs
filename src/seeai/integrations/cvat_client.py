import json
import os
import shutil
from pathlib import Path
from cvat_sdk import make_client
from cvat_sdk.api_client import models
from src.seeai.config.settings import CVAT_HOST_IP, CVAT_HOST_PORT, CVAT_USER, CVAT_PASSWORD
from src.seeai.utils.file_utils import extract_and_flatten_zip
from src.seeai.data.dataset_manager import remove_dataset_files

CVAT_HOST = f"{CVAT_HOST_IP}:{CVAT_HOST_PORT}"

def create_cvat_project_with_tasks(datasets_dir, selected_dataset_name):
    """
    Create a CVAT project and split dataset into Train/Val/Test tasks (7:1:2 ratio).
    """
    url = CVAT_HOST
    username = CVAT_USER
    password = CVAT_PASSWORD
    org_id = 1
    
    if not all([url, username, password]):
        return "❌ Error: Missing CVAT credentials."
        
    # Sanitize URL and ensure it has http:// protocol
    url = url.split('/projects')[0].split('/tasks')[0].split('jobs')[0].rstrip('/')
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    print(f"CVAT URL: {url}")

    dataset_name = selected_dataset_name
    if not dataset_name:
        return "❌ Error: No dataset selected."

    # Locate the inference output directory
    output_dir = Path(datasets_dir)/ '.output' / f"{dataset_name}_coco"
    images_dir = output_dir / "images" / "Train"
    annotations_file = output_dir / "annotations" / "instances_Train.json"

    if not images_dir.exists() or not annotations_file.exists():
        return f"❌ Error: Inference output not found at {output_dir}. Please run 'Inference Dataset' first."

    try:
        with make_client(url, credentials=(username, password)) as client:
            
            # 1. Load COCO data
            print("📖 Loading COCO annotations...")
            with open(annotations_file, 'r') as f:
                coco_data = json.load(f)
            
            images = coco_data.get('images', [])
            annotations = coco_data.get('annotations', [])
            categories = coco_data.get('categories', [])
            inf_format = coco_data.get('info', {}).get('inference_format', 'Detection')
            
            # 2. Split dataset (7:2:1 ratio)
            print(f"🔀 Splitting {len(images)} images into Train/Val/Test (7:2:1 ratio)...")
            import random
            random.seed(42)
            shuffled_images = random.sample(images, len(images))
            
            n_total = len(shuffled_images)
            if n_total < 2:
                train_images = shuffled_images
                val_images = []
                test_images = []
                print(f"  ⚠️ Warning: Only {n_total} image(s). Val and Test will be empty.")
            elif n_total == 2:
                train_images = shuffled_images[:1]
                val_images = shuffled_images[1:2]
                test_images = []
            else:
                n_train = max(1, int(n_total * 0.7))
                n_val = max(1, int(n_total * 0.2)) 
                n_test = n_total - n_train - n_val
                if n_test <= 0:
                    n_test = 0
                    if n_total == 3:
                        n_train = 2
                        n_val = 1
                    else:
                        n_val = max(1, n_total // 3)
                        n_train = n_total - n_val
                
                train_images = shuffled_images[:n_train]
                val_images = shuffled_images[n_train:n_train+n_val]
                test_images = shuffled_images[n_train+n_val:n_train+n_val+n_test]
                
            print(f"  📊 Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
            
            train_ids = {img['id'] for img in train_images}
            val_ids = {img['id'] for img in val_images}
            test_ids = {img['id'] for img in test_images}
            
            # 3. Create CVAT Project
            project_name = f"{dataset_name}_project"
            print(f"🏗️ Creating CVAT project: {project_name}")
            
            labels = [
                models.PatchedLabelRequest(
                    name=cat['name'],
                    color="#ff0000",
                    attributes=[]
                ) 
                for cat in categories
            ]
            
            project_spec = models.ProjectWriteRequest(
                name=project_name,
                labels=labels
            )
            
            if org_id:
                project_data, _ = client.api_client.projects_api.create(
                    project_write_request=project_spec, 
                    org_id=org_id
                )
            else:
                project_data, _ = client.api_client.projects_api.create(
                    project_write_request=project_spec
                )
            project_id = project_data.id
            print(f"  ✅ Project created (ID: {project_id})")
            
            # 4. Create 3 tasks
            if inf_format == 'Classification':
                subsets = [("train", train_images, train_ids), ("val", val_images, val_ids), ("test", test_images, test_ids)]
            else:
                subsets = [("Train", train_images, train_ids), ("Validation", val_images, val_ids), ("Test", test_images, test_ids)]
            
            task_urls = []
            
            for subset_name, subset_images, subset_ids in subsets:
                if len(subset_images) == 0:
                    continue
                
                print(f"\n📝 Creating task: {dataset_name}_{subset_name}")
                task_spec = models.TaskWriteRequest(
                    name=f"{dataset_name}_{subset_name}",
                    project_id=project_id,
                    subset=subset_name,
                    segment_size=0
                )
                
                if org_id:
                    task_data, _ = client.api_client.tasks_api.create(task_write_request=task_spec, org_id=org_id)
                else:
                    task_data, _ = client.api_client.tasks_api.create(task_write_request=task_spec)
                
                print(f"  ✓ Task created (ID: {task_data.id})")
                high_level_task = client.tasks.retrieve(task_data.id)    
                
                print(f"  📤 Uploading {len(subset_images)} images...")
                image_files = [str(images_dir / img['file_name']) for img in subset_images]
                high_level_task.upload_data(image_files)
                
                subset_annotations = [ann for ann in annotations if ann['image_id'] in subset_ids]
                
                temp_coco = {
                    'images': subset_images,
                    'annotations': subset_annotations,
                    'categories': categories,
                    'info': coco_data.get('info', {})
                }
                
                temp_coco_file = output_dir / f"temp_{subset_name}.json"
                with open(temp_coco_file, 'w') as f:
                    json.dump(temp_coco, f)
                
                print(f"  📥 Importing {len(subset_annotations)} annotations...")
                high_level_task.import_annotations(format_name="COCO 1.0", filename=str(temp_coco_file))
                
                # Classification tags
                if inf_format == "Classification":
                    try:
                        print(f"  🏷️  Applying tag annotations for Classification...")
                        task_labels = high_level_task.get_labels()
                        label_name_to_id = {l.name: l.id for l in task_labels}
                        sorted_subset_images = sorted(subset_images, key=lambda x: x['file_name'])
                        image_id_to_frame = {img_info['id']: idx for idx, img_info in enumerate(sorted_subset_images)}
                        category_map = {c['id']: c['name'] for c in categories}
                        tags_to_create = []
                        
                        for ann in subset_annotations:
                            img_id = ann['image_id']
                            cat_id = ann['category_id']
                            if img_id not in image_id_to_frame: continue
                            frame_idx = image_id_to_frame[img_id]
                            cat_name = category_map.get(cat_id)
                            if not cat_name or cat_name.strip() == '': continue
                            if ' ' in cat_name: continue
                            if cat_name in label_name_to_id:
                                cvat_label_id = int(label_name_to_id[cat_name])
                                tags_to_create.append(models.LabeledImageRequest(frame=int(frame_idx), label_id=cvat_label_id))
                        
                        if tags_to_create:
                            jobs = high_level_task.get_jobs()
                            if jobs:
                                from types import SimpleNamespace
                                jobs[0].update_annotations(models.PatchedLabeledDataRequest(tags=tags_to_create), action=SimpleNamespace(value="create"))
                                print(f"    ✅ Created {len(tags_to_create)} tag annotations")
                    except Exception as e:
                        print(f"    ⚠️ Warning: Tag annotation failed: {e}")
                
                temp_coco_file.unlink()
                task_url = f"{url.rstrip('/')}/tasks/{task_data.id}"
                task_urls.append((subset_name, task_data.id, task_url))
                print(f"  ✅ Task complete: {task_url}")
            
            # Update subsets
            print("\n🔄 Updating task subsets based on names...")
            paginated_data, _ = client.api_client.tasks_api.list(project_id=int(project_id))
            tasks = paginated_data.results
            for task in tasks:
                subset_name = task.subset
                if not subset_name: continue
                subset_lower = subset_name.lower()
                current_subset = None
                if "train" in subset_lower: current_subset = "Train"
                elif "val" in subset_lower or "valid" in subset_lower: current_subset = "Validation"
                elif "test" in subset_lower: current_subset = "Test"
                
                if current_subset and current_subset.lower() != subset_name.lower():
                     client.api_client.tasks_api.partial_update(id=task.id, patched_task_write_request=models.PatchedTaskWriteRequest(subset=current_subset))
            
            print("✅ Task subsets updated successfully.")
            print("\n🧹 Cleaning up temporary files...")
            shutil.rmtree(output_dir.parent, ignore_errors=True)
            
            # Remove local dataset
            # Determine path - datasets_dir / dataset_name ??
            # `dataset_name` is just name. Path logic needed.
            # Assuming standard structure: datasets_dir / dataset_name (if directory)
            # We call remove_dataset_files which takes path.
            # Need to find path first.
            possible_path = Path(datasets_dir) / dataset_name
            if possible_path.exists():
                remove_dataset_files(str(possible_path))
            
            project_url = f"{url.rstrip('/')}/projects/{project_id}"
            
            tasks_html = "<br>".join([
                f"<b>{name}:</b> <a href='{task_url}' target='_blank' style='color: var(--link-text-color); text-decoration: underline;'>Task {task_id}</a>"
                for name, task_id, task_url in task_urls
            ])
            
            return (
                f"<div style='padding: var(--size-2); border: 1px solid var(--block-border-color); "
                f"background: var(--input-background-fill); border-radius: var(--container-radius); "
                f"color: var(--body-text-color); min-height: 120px;'>"
                f"<strong>✅ CVAT Project Created Successfully!</strong><br><br>"
                f"<b>Project:</b> <a href='{project_url}' target='_blank' style='color: var(--link-text-color); text-decoration: underline;'>{project_name} (ID: {project_id})</a><br><br>"
                f"<b>Tasks Created:</b><br>{tasks_html}"
                f"</div>"
            )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error creating CVAT project: {str(e)}"

def upload_to_cvat(datasets_dir, zip_file_path, dataset_format, progress=None):
    """
    Upload a dataset zipfile that had already been formatted following a standard format to CVAT.
    """
    if not zip_file_path or not os.path.exists(zip_file_path):
        return False, "❌ No zip file provided"
    
    cvat_format = dataset_format
    organization = "PixeVision"
    
    try:
        # Step 1: Flatten
        if progress: progress(0.0, desc="📦 Extracting and flattening dataset...")
        dataset_name = Path(zip_file_path).stem
        temp_dir = Path(datasets_dir) / f"temp_{dataset_name}"
        
        extract_and_flatten_zip(zip_file_path, str(temp_dir))
        
        # Step 2: Re-zip
        if progress: progress(0.25, desc="📁 Creating zip archive...")
        flattened_zip = Path(datasets_dir) / f"{dataset_name}_flattened.zip"
        shutil.make_archive(str(flattened_zip.with_suffix('')), 'zip', str(temp_dir))

        if temp_dir.exists(): shutil.rmtree(temp_dir)
        
        # Step 3: CVAT Connect
        if progress: progress(0.50, desc="🔗 Connecting to CVAT...")
        url = CVAT_HOST
        username = CVAT_USER
        password = CVAT_PASSWORD
        
        if not all([url, username, password]):
            return False, "❌ Missing CVAT credentials"
        
        host = url.split('/projects')[0].split('/tasks')[0].split('/jobs')[0].rstrip('/')
        
        with make_client(host, credentials=(username, password)) as client:
            if progress: progress(0.60, desc="📁 Creating CVAT project...")
            project_spec = models.ProjectWriteRequest(name=dataset_name)
            
            if organization:
                (project_data, _) = client.api_client.projects_api.create(project_spec, org=organization)
            else:
                (project_data, _) = client.api_client.projects_api.create(project_spec)
                
            project_id = project_data.id if hasattr(project_data, 'id') else (project_data['id'] if isinstance(project_data, dict) else None)
            
            if project_id is None:
                return False, "❌ Error: CVAT Project ID is None."

            project = client.projects.retrieve(int(project_id))
            
            if progress: progress(0.70, desc="📥 Uploading dataset to CVAT...")
            project.import_dataset(format_name=cvat_format, filename=str(flattened_zip))
            
            if progress: progress(0.85, desc="🔄 Updating task subsets...")
            
            paginated_data, _ = client.api_client.tasks_api.list(project_id=int(project_id))
            tasks = paginated_data.results
            for task in tasks:
                subset_name = task.subset
                current_subset = None
                if "train" in subset_name: current_subset = "Train"
                elif "val" in subset_name or "valid" in subset_name: current_subset = "Validation"
                elif "test" in subset_name: current_subset = "Test"
                
                if current_subset:
                   if organization:
                        client.api_client.tasks_api.partial_update(id=task.id, patched_task_write_request=models.PatchedTaskWriteRequest(subset=current_subset))
            
        if flattened_zip.exists(): os.remove(flattened_zip)
        
        return True, f"Successfully uploaded {dataset_name} to CVAT (Project ID: {project_id})"
            
    except Exception as e:
        print(f"Error uploading to CVAT: {e}")
        return False, f"Error: {e}"
