import gradio as gr
import os
import zipfile
import tempfile
import shutil
import cv2

def is_video(filename):
    return filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))

def is_image(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))

def handle_upload(zip_file):
    if zip_file is None:
        return gr.update(visible=False), gr.update(visible=False), "Please upload a zip file.", None, []
        
    temp_dir = tempfile.mkdtemp()
    
    try:
        with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    except (zipfile.BadZipFile, EOFError, Exception) as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return gr.update(visible=False), gr.update(visible=False), f"Invalid or incomplete zip file uploaded. Error: {str(e)}", None, []
        
    videos = []
    images = []
    
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if is_video(file):
                videos.append(os.path.join(root, file))
            elif is_image(file):
                images.append(os.path.join(root, file))
                
    if not videos and not images:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return gr.update(visible=False), gr.update(visible=False), "No videos or images found in the zip file.", None, []
        
    status = f"**Status:** Found {len(videos)} videos and {len(images)} images."
    
    has_videos = len(videos) > 0
    
    return gr.update(visible=has_videos), gr.update(visible=True), status, temp_dir, videos

def process_dataset(temp_dir, videos_list, interval_sec):
    if not temp_dir or not os.path.exists(temp_dir):
        return None, "No valid dataset to process. Please upload again."
        
    final_output_dir = tempfile.mkdtemp()
    
    # Copy images
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if is_image(file):
                src = os.path.join(root, file)
                rel_path = os.path.relpath(root, temp_dir)
                if rel_path == '.':
                    dst_dir = final_output_dir
                else:
                    dst_dir = os.path.join(final_output_dir, rel_path)
                    os.makedirs(dst_dir, exist_ok=True)
                    
                dst = os.path.join(dst_dir, file)
                try:
                    shutil.copy2(src, dst)
                except:
                    pass
                    
    # Process videos
    if videos_list:
        try:
            val = float(interval_sec)
        except:
            val = 1.0
            
        for video_path in videos_list:
            video_name = os.path.basename(video_path)
            
            # Place extracted frames in the same relative path structure
            rel_path = os.path.relpath(os.path.dirname(video_path), temp_dir)
            if rel_path == '.':
                dst_dir = final_output_dir
            else:
                dst_dir = os.path.join(final_output_dir, rel_path)
                os.makedirs(dst_dir, exist_ok=True)
                
            try:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0: 
                    cap.release()
                    continue
                
                step_frames = int(fps * val)
                if step_frames < 1: 
                    step_frames = 1
                
                frame_idx = 0
                count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    if frame_idx % step_frames == 0:
                        frame_name = f"{os.path.splitext(video_name)[0]}_frame_{count:04d}.jpg"
                        cv2.imwrite(os.path.join(dst_dir, frame_name), frame)
                        count += 1
                        
                    frame_idx += 1
                cap.release()
            except Exception as e:
                print(f"Failed to process video {video_name}: {e}")
                
    # Zip output
    output_zip_path = os.path.join(tempfile.mkdtemp(), "processed_dataset.zip")
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(final_output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, final_output_dir)
                zipf.write(file_path, arcname)
                
    # Cleanup temp dirs
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(final_output_dir, ignore_errors=True)
    
    return output_zip_path, "Dataset processed and zipped successfully!"

with gr.Blocks(title="Video to Dataset") as app:
    gr.Markdown("# Video to Dataset Converter")
    gr.Markdown("Upload a ZIP file containing images and/or videos. If videos are present, you can adjust the extraction interval to convert them into image frames.")
    
    with gr.Row():
        with gr.Column():
            zip_upload = gr.File(label="Upload Dataset ZIP", file_types=[".zip"])
            status_text = gr.Markdown()
            
            with gr.Group(visible=False) as video_controls:
                interval_slider = gr.Slider(minimum=0.1, maximum=10.0, value=1.0, step=0.1, label="Extraction Interval (seconds)")
                
            process_btn = gr.Button("Process Dataset", visible=False, variant="primary")
            
        with gr.Column():
            output_file = gr.File(label="Download Processed Dataset ZIP", interactive=False)
            result_text = gr.Markdown()
            
    # Hidden states to store paths
    temp_dir_state = gr.State(None)
    videos_list_state = gr.State([])
    
    zip_upload.upload(
        handle_upload,
        inputs=[zip_upload],
        outputs=[video_controls, process_btn, status_text, temp_dir_state, videos_list_state]
    )
    
    process_btn.click(
        process_dataset,
        inputs=[temp_dir_state, videos_list_state, interval_slider],
        outputs=[output_file, result_text]
    )

if __name__ == "__main__":
    app.launch()
