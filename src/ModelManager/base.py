import io
import json

from pathlib import Path

import cv2
import torch
import numpy as np
import litserve as ls

from ultralytics import YOLO
from ultralytics.engine.results import Results

from .utils import jsonify_ultra_result

class YOLOAI(ls.LitAPI):
    def __init__(self,
        model_path: str | Path = Path('./assests/best.pt'),
        result_atts: list = None,
        **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.result_atts = result_atts
    
    def setup(self, device):
        self.device = device
        self.model = YOLO(self.model_path)

    async def decode_request(self, request) -> list[np.ndarray]:
        rq_stream = await request['stream'].read()
        rq_meta = await request['meta'].read()
        rq_meta = json.loads(rq_meta.decode('utf-8'))
        
        if rq_meta.get('preprocessed', False) == False:
            x = np.frombuffer(bytearray(rq_stream), dtype=np.uint8)
            images = [cv2.imdecode(x, cv2.IMREAD_COLOR_BGR)]
        
        else:
            x = np.load(io.BytesIO(rq_stream), allow_pickle=False)
            images = torch.from_numpy(x).to(self.device)
        
        return images

    async def predict(self, x: list[np.ndarray]) -> list[Results]:
        result = self.model.predict(
            x,
            verbose=False,
            device=self.device,
            save=False,
            )
        return result

    async def encode_response(self, output: list[Results]) -> dict:
        out = output[0]
        iterator = self.result_atts \
            if self.result_atts is not None \
            else [x for x in out.__dict__.keys() if not x.startswith("_")]
        result = dict()
        iterator.remove('orig_img')
        iterator.remove('path')
        iterator.remove('orig_shape')
        for att in iterator:
            value = getattr(out, att)
            if value is not None:
                result[att] = jsonify_ultra_result(value)
        return result