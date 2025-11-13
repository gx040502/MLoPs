import io
import os
import json
import requests

from pathlib import Path

import cv2
import numpy as np
import gradio as gr

from dotenv import load_dotenv
from ultralytics.utils.plotting import Colors, Annotator

from server.utils import BasePreprocessor

# Global state
load_dotenv(Path(__file__).parents[1].joinpath('.env'))
url = os.getenv('nasivision-ai-api-url')
token = os.getenv('nasivision-ai-api-token')
prep = BasePreprocessor()
colors = Colors()



def request_detection(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    orig_shape = image.shape
    img = prep([image])
    a = io.BytesIO()
    np.save(a, img, allow_pickle=False)
    binary_data = a.getvalue()

    response = requests.post(url, files={
        'stream': binary_data,
        'meta': json.dumps({'preprocessed': True}), 
        }, headers={'Authorization': f"Bearer {token}"}
        )
    
    data = response.json()
    names = data['names']
    
    # If preprocessed/resized image is sent to server
    xyxy = np.array(data['boxes']['xyxyn'])
    if not len(xyxy) < 1: 
        xyxy[:, ::2] *= orig_shape[1]
        xyxy[:, 1::2] *= orig_shape[0]
    xyxy = np.rint(xyxy).astype(np.intp)

    cls = np.array(data['boxes']['cls'], dtype=np.intp).astype(str).tolist()
    conf = np.array(data['boxes']['conf']).tolist()

    anno = Annotator(image.copy())
    for bx, cls_idx, score in zip(xyxy, cls, conf):
        anno.box_label(
            box=bx,
            label=f"{names[cls_idx]}: {score:0.4f}",
            color=colors(cls_idx),
            txt_color=colors(cls_idx)[::-1],
            )
    
    im = anno.im
    # resized = cv2.resize(im, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    resized = im
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    return resized

def capNdetect(frame):
    return request_detection(frame)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 Capture and Detect Separately")

    with gr.Column():
        with gr.Row():
            webcam_input = gr.Image(
                label="Live Camera Input",
                sources=['webcam'],
                type='numpy',
                interactive=True,
                )
            detection_view = gr.Image(label="Detection Result")

        # with gr.Column():
        #     capNdetect_btn = gr.Button("🔍 Detect")
    
    webcam_input.change(fn=capNdetect, inputs=webcam_input, outputs=detection_view)
    # capNdetect_btn.click(fn=capNdetect, inputs=webcam_input, outputs=detection_view)

if __name__ == '__main__':
    demo.launch(share=True)









