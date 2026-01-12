import os
import cv2
import shutil
import zipfile
import pandas as pd

def scan_for_videos(dataset_path):
    """
    Scans the dataset for video files and returns a list of dictionaries.
    Each dictionary contains: 'filename', 'duration', 'duration_sec'.
    """
    if not dataset_path or not os.path.exists(dataset_path):
        return []
        
    video_files = []
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    
    for file in os.listdir(dataset_path):
        if file.lower().endswith(video_extensions):
            full_path = os.path.join(dataset_path, file)
            try:
                cap = cv2.VideoCapture(full_path)
                if not cap.isOpened():
                    continue
                    
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_sec = frame_count / fps if fps > 0 else 0
                
                # specific formatting for duration
                minutes = int(duration_sec // 60)
                seconds = int(duration_sec % 60)
                duration_str = f"{minutes}m {seconds}s"
                
                video_files.append({
                    "Video Name": file,
                    "Duration": duration_str,
                    "Extraction Interval": "1s" # Default value
                })
                cap.release()
            except Exception as e:
                print(f"Error reading video {file}: {e}")
                
    return video_files

def extract_frames_from_dataset(dataset_path, dataset_name, datasets_dir, video_config_df, interval_val=1.0):
    """
    Creates a temporary dataset with extracted frames + original images.
    
    Args:
        dataset_path: Path to the source dataset directory
        dataset_name: Name of the source dataset
        datasets_dir: Root directory where datasets are stored (output loc)
        video_config_df: pandas DataFrame or list containing video info
        interval_val: float (seconds)
        
    Returns:
        (success, result) - result is zip_output_path if success else error message
    """
    if not dataset_path or not os.path.exists(dataset_path):
        return False, "Source dataset path not valid"

    temp_name = f"{dataset_name}_temp_frames"
    temp_dir = os.path.join(datasets_dir, temp_name)
    
    # 1. Create temp directory (clean if exists)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    # 2. Copy existing images
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    for file in os.listdir(dataset_path):
        if file.lower().endswith(image_extensions):
            shutil.copy2(os.path.join(dataset_path, file), os.path.join(temp_dir, file))
            
    # 3. Process Videos
    # Convert df to list of dicts if needed (pandas df to records)
    if isinstance(video_config_df, pd.DataFrame):
        config_records = video_config_df.to_dict('records')
    else:
        if isinstance(video_config_df, list):
            if len(video_config_df) > 0 and isinstance(video_config_df[0], list):
                    config_records = [
                        {"Video Name": r[0], "Duration": r[1]} 
                        for r in video_config_df
                    ]
            else:
                config_records = video_config_df
        else:
                config_records = []

    # Calculate target interval in seconds
    try:
        val = float(interval_val)
    except:
        val = 1.0
        
    target_interval_sec = val

    for record in config_records:
        video_name = record.get("Video Name")
        interval_sec = target_interval_sec
        
        video_path = os.path.join(dataset_path, video_name)
        if not os.path.exists(video_path):
            continue
            
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: continue
            
            step_frames = int(fps * interval_sec)
            if step_frames < 1: step_frames = 1
            
            frame_idx = 0
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % step_frames == 0:
                    # Save frame
                    frame_name = f"{os.path.splitext(video_name)[0]}_frame_{count:04d}.jpg"
                    cv2.imwrite(os.path.join(temp_dir, frame_name), frame)
                    count += 1
                    
                frame_idx += 1
            cap.release()
        except Exception as e:
            print(f"Failed to extract {video_name}: {e}")

    # 4. Zip the new dataset so it can be re-uploaded or used as main source
    # We need to zip the contents of temp_dir
    zip_output_path = os.path.join(datasets_dir, f"{temp_name}.zip")
    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)
                
    return True, zip_output_path
