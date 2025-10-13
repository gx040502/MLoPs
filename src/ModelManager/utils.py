import time
import json
import yaml
import shutil
import zipfile

from pathlib import Path
from http import HTTPStatus
from urllib.parse import parse_qsl, urlparse

from tqdm import tqdm
from ultralytics import settings
from cvat_sdk.api_client import Configuration, ApiClient, models

def download_cvat_dataset(
    config: Configuration = None,
    task_id: int = None,
    project_id: int = None,
    format_name: str = "Ultralytics YOLO Detection 1.0",
    include_images: bool = True,
    output_path: str | Path = None,
    poll_interval: float = 1,
    max_retries: int = 100,
    chunk_size: int = 8192, # 8192 Bytes = 8KB
) -> tuple[bool, Path]:
    """Download CVAT dataset with specfied format in zip file.
    
    Args:
        config (Configuration):
            `Configuration` instance from cvat_sdk.api_client.
        task_id (int):
            Task `ID` number according to CVAT.
        project_id (int):
            Project `ID` number according to CVAT.
        format_name (str):
            Desired output format name.
            You can get the list of supported formats at
            https://docs.cvat.ai/docs/manual/advanced/formats/.
        include_images (bool):
            Whether to save image to zip file.
        output_path (str | Path):
            Path to save the downloaded zipfile.
        poll_interval (int):
            Seconds to wait before the next retry to retrieve dataset 
            preparation status.
        max_retries (int):
            Max number of retries before giving up on waiting for the dataset 
            preparation.
        chunk_size (int): 
            Number of bytes per chunk in the zip file download. 
            Default is 8192 Bytes = 8KB.

    Returns:
        (tuple[bool, Path])
            - bool: The existence of the zip file.
            - Path: The `Path` object to the zip file

    Raises:
        ValueError: Cannot decide the dataset is task or project level dataset
            from just the ids.
        TimeoutError: Dataset preparation take longer than the retry interval.
    """
    if not any([task_id, project_id]):
        raise ValueError("You must provide either task_id or project_id.")
    
    if all([task_id, project_id]):
        raise ValueError("You must provide only either task_id or project_id.")

    if output_path is None:
        if task_id:
            output_path = Path(f'task_{task_id}'). resolve()
        else:
            output_path = Path(f'project_{project_id}').resolve()
    
    else:
        output_path = Path(output_path)
            
    with ApiClient(config) as client:
        if task_id:
            _, response = client.tasks_api.create_dataset_export(
                id=task_id,
                format=format_name,
                save_images=include_images,
            )
        else:
            _, response = client.projects_api.create_dataset_export(
                format=format_name,
                id=project_id,
                save_images=include_images,
            )

        rq_id = json.loads(response.data).get("rq_id")
        assert rq_id, "The rq_id parameter was not found in the server response"
        
        # Check the dataset preparation progress
        for _ in tqdm(
            range(max_retries),
            total=max_retries-1,
            desc="Waiting Dataset Export Preparation"):
            (rq_status, response) = client.requests_api.retrieve(rq_id)
            assert response.status == HTTPStatus.OK
            process_status = rq_status.status.value
            if process_status in (
                models.RequestStatus.allowed_values[("value",)]["FINISHED"],
                models.RequestStatus.allowed_values[("value",)]["FAILED"],
            ):
                break
            time.sleep(poll_interval)
        
        # Check the dataset preparation status after retries
        if process_status != models.RequestStatus.allowed_values[("value",)]["FINISHED"]:
            exception_msg = f"Export failed: {process_status}"
            if rq_status.message:
                exception_msg += f". Details: {rq_status.message}"
            raise TimeoutError(f"Export failed: {process_status}")
        
        # Download the prepared file
        download_url = rq_status.result_url
        assert download_url, "No 'download_url' in the server response"
        
        parsed_download_url = urlparse(download_url)
        query_params = parse_qsl(parsed_download_url.query)
        
        _, response = client.call_api(
            parsed_download_url.path,
            method="GET",
            query_params=query_params,
            auth_settings=client.configuration.auth_settings(),
            _parse_response=False,
        )
           
        with open(output_path, "wb") as output_file, tqdm(
            total=int(response.headers.get('Content-Length')),
            unit='B',
            unit_scale=True,
            desc=output_path.as_posix()
        ) as pbar:
            while (chunk := response.read(chunk_size)):
                output_file.write(chunk)
                pbar.update(len(chunk))

    return output_path.exists(), output_path

def extract_cvat_zip(zip_path: str) -> Path:
    '''Extract CVAT exported Zip file and modify for Ultralytics training.
    
    Args:
        zip_path (str): The zip file, e.g. './datasets/coco8.yaml' will be 
            extracted to './datasets/coco8/'.
    
    Returns:
        yaml_file (Path): The path to the `data.yaml` file.
    
    Raises:
        ValueError:
            - `zip_path` entered is not a zip file.
            - `path` and `names` missing in the `data.yaml` file.
        FileNotFoundError:
            - `data.yaml` not in zip file root directory.
            - CVAT uses text file to record relative path to each image, and one
            of the images is missing.
        zipfile.BadZipFile: Zip file is corrupted.
    '''
    zip_path = Path(zip_path)
    
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("❌ Not a valid ZIP file.")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        bad_file = zip_ref.testzip()
        if bad_file:
            raise zipfile.BadZipFile(f"❌ Corruption detected in: {bad_file}")

    root = zip_path.parent / zip_path.stem
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(zip_path.parent / zip_path.stem)

    yaml_file = root / "data.yaml"
    if not yaml_file.exists():
        raise FileNotFoundError("❌ data.yaml missing.")

    # Verify yaml content
    yaml_content = yaml.safe_load(open(yaml_file))
    gt_yaml = {**yaml_content}
    if any([x not in yaml_content.keys() for x in ('path', 'names')]):
        raise ValueError(f"❌ 'path' or 'names' not in {yaml_file} file")
    
    # 'path' should be relative to dataset root directory in Ultralytics's
    # 'datasets_dir' settings
    ultralytics_dataset_dir = Path(settings.get('datasets_dir'))
    if Path(ultralytics_dataset_dir) / gt_yaml['path'] != root:
        gt_yaml['path'] = root.relative_to(ultralytics_dataset_dir).as_posix()
    
    if 'train' not in yaml_content.keys():
        gt_yaml['train'] = yaml_content.get('Train')
        del gt_yaml['Train']
    
    if 'val' not in yaml_content.keys():
        gt_yaml['val'] = yaml_content.get('Validation')
        del gt_yaml['Validation']
    
    # Rename original file for future reference and safe the updated configuration
    # with the default 'data.yaml' name
    shutil.move(yaml_file, yaml_file.parent / (yaml_file.stem + '_ori' + yaml_file.suffix))
    yaml.dump(gt_yaml, open(yaml_file, 'w'))
    
    for split in ["train", "val"]:
        if '.txt' in gt_yaml[split]:
            split_txt = ultralytics_dataset_dir / gt_yaml['path'] / gt_yaml[split]
            with open(split_txt) as f:
                txt_list = f.readlines()
            
            img_list = [x.strip() for x in txt_list]
            if any([x.startswith('data/') for x in img_list]):
                gt_img_list = []
                for x in img_list:
                    # Must replace with './' if start with 'images/train/xxx.jpg'
                    # will cause bug when changing to 'labels/train/xxx.txt'
                    # should be './images/train/xxx.jpg' relative to 'path' in
                    # data.yaml
                    if x.startswith('data/'):
                        new_x = './' + x.removeprefix('data/')
                        
                    else:
                        new_x = x
                    gt_img_list.append(new_x)
            
            # The inidividual relvative path in the .txt file for 'train' and
            # 'val' should be relative to the 'path' in the 'data.yaml' file 
            shutil.move(split_txt, split_txt.parent / (split_txt.stem + '_ori' + split_txt.suffix))
            with open(split_txt, 'w') as f:
                for x in gt_img_list:
                    f.writelines(x + '\n')
            
            img_list = [
                ultralytics_dataset_dir / gt_yaml['path'] / x for x in img_list
            ]
            
            missing_file_boolean = [not x.exists() for x in img_list]
            if any((not x for x in missing_file_boolean)):
                raise FileNotFoundError(
                    f"❌ Missing images that are stated in {split_txt}: \
                    {[item for item, keep in zip(img_list, missing_file_boolean) if keep]}"
                    )
    
    return yaml_file
