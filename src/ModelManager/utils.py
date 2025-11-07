import time
import json
import yaml
import zipfile

from pathlib import Path
from http import HTTPStatus
from urllib.parse import parse_qsl, urlparse

import torch
import numpy as np

from tqdm import tqdm
from ultralytics import settings
from ultralytics.engine.results import Boxes, Masks, Probs
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
        
        exist_size = output_path.stat().st_size if output_path.exists() else 0
        try:
            _, response = client.call_api(
                parsed_download_url.path,
                method="GET",
                query_params=query_params,
                header_params={"Range": f"bytes={exist_size}-"},
                auth_settings=client.configuration.auth_settings(),
                _parse_response=False,
            )
        
            total_size = int(response.headers.get('Content-Range').split('/')[-1])
            
            with open(output_path, "wb" if exist_size == 0 else "ab") as output_file, tqdm(
                initial=exist_size,
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=output_path.as_posix()
            ) as pbar:
                while pbar.n < total_size:
                    chunk = response.read(chunk_size)
                    output_file.write(chunk)
                    pbar.update(len(chunk))
        
        except Exception as e:
            # if file exist and size equals to full download size
            if e.status == HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE:
                pass
            else:
                raise e

    return output_path.exists(), output_path

def extract_cvat_zip(
    zip_path: str | Path,
    dst_dir: str | Path = Path(settings.get('datasets_dir')),
    dataset_name: str = None,
    ) -> Path:
    '''Extract CVAT exported Zip file and modify for Ultralytics training.
    For example, for dataset `project_#1` with path of './project_#1.zip', will
    be extracted to Path(f"{`dst_dir`}/{`dataset_name`}").

    Args:
        zip_path (str | Path): Path to the zip file.
        dst_dir (str | Path): The destination directory root directory excluding
            the name of the dataset.
            Default is the `datasets_dir` from ultralytic's `settings` for the
            current user.
        dataset_name (str): Name of the dataset, if not provided the stem of zip
            file will be used.
            Default is the `stem` of the `zip_path`.
    
    Returns:
        yaml_file (Path): The path to the `data.yaml` file.
    
    Raises:
        ValueError:
            - `zip_path` entered is not a zip file.
        FileNotFoundError:
            - `data.yaml` not in zip file root directory.
            - CVAT uses text file to record relative path to each image, and one
            of the images is missing.
        zipfile.BadZipFile:
            - Zip file is corrupted.
    '''
    zip_path = Path(zip_path)
    dst = Path(dst_dir)
    if dataset_name is None:
        dataset_name = zip_path.stem
    
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("❌ Not a valid ZIP file.")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        bad_file = zip_ref.testzip()
        if bad_file:
            raise zipfile.BadZipFile(f"❌ Corruption detected in: {bad_file}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        for m in tqdm(members, desc='Extracting', unit='file'):
            zip_ref.extract(member=m, path=dst / dataset_name)
    
    yaml_file = dst / dataset_name / "data.yaml"
    if not yaml_file.exists():
        raise FileNotFoundError("❌ data.yaml missing.")

    # Verify yaml content
    yaml_content = yaml.safe_load(open(yaml_file))
    gt_yaml = {**yaml_content}
    if dst / gt_yaml['path'] != dst / dataset_name:
        gt_yaml['path'] = (dst / dataset_name).relative_to(dst).as_posix()
    
    data_trainval = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}
    for new, old in data_trainval.items():
        if new not in yaml_content.keys():
            gt_yaml[new] = yaml_content.get(old)
        
        if gt_yaml.get(old) is not None:
            del gt_yaml[old]
    
    # Rename original file for future reference and safe the updated configuration
    # with the default 'data.yaml' name
    # shutil.move(yaml_file, yaml_file.parent / (yaml_file.stem + '_ori' + yaml_file.suffix))
    gt_yaml = {k: v for k,v in gt_yaml.items() if v is not None}
    yaml.dump(gt_yaml, open(yaml_file, 'w'))
    
    for split in data_trainval.keys():
        if '.txt' in gt_yaml.get(split, ''):
            split_txt = dst / gt_yaml['path'] / gt_yaml[split]
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
            # shutil.move(split_txt, split_txt.parent / (split_txt.stem + '_ori' + split_txt.suffix))
            with open(split_txt, 'w') as f:
                for x in gt_img_list:
                    f.writelines(x + '\n')
            
            img_list = [
                dst / gt_yaml['path'] / x for x in img_list
            ]
            
            missing_file_boolean = [not x.exists() for x in img_list]
            if any((not x for x in missing_file_boolean)):
                raise FileNotFoundError(
                    f"❌ Missing images that are stated in {split_txt}: \
                    {[item for item, keep in zip(img_list, missing_file_boolean) if keep]}"
                    )
    
    return yaml_file

def jsonify_ultra_result(result: Boxes | Masks | Probs) -> dict:
    '''Handle each item in ultralytics.engine.results.Result instance to make 
    it json compatible because numpy.ndarray and torch.Tensor is not uncompatible
    with json.
    
    Args:
        result (Boxes | Masks | Probs): Any one of the attributes in
            ultralytics.engine.results.Result class
    
    Returns:
        res (dict): A dictionary or any data type or data structure that is json
            compatible.
    
    '''
    if isinstance(result, (Boxes, Masks, Probs)):
        attributes = [x for x in result.__class__.__dict__.keys() if not x.startswith("__")]
        res = {}
        for k in attributes:
            value = getattr(result, k)
            res[k] = value.tolist() if isinstance(value, (torch.Tensor, np.ndarray)) else value
    elif isinstance(result, (torch.Tensor, np.ndarray)):
        res = result.tolist()
    elif isinstance(result, tuple):
        res = list(result)
    else:
        res = result
    return res