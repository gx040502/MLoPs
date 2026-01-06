import io
import json
import pytest

import cv2
import torch
import numpy as np
import litserve as ls

from fastapi.testclient import TestClient
from litserve.utils import wrap_litserve_start

from ModelManager.base import YOLOAI # from server.base import YOLOAI
from ModelManager.utils import BasePreprocessor # from server.utils import BasePreprocessor

# Prepare da
sample = './assests/zapfan.jpeg'
buffer = open(sample, 'rb')
img = cv2.imread(sample)
prep = BasePreprocessor()
ndarray = prep([img])
a = io.BytesIO()
np.save(a, ndarray, allow_pickle=False)
binary_data = a.getvalue()

def test_torch_cpu():
    #create temporary instance if the AI server on the CPU
    server = ls.LitServer(YOLOAI(enable_async=True), accelerator="cpu", devices=1)
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        # Unprocessed image
        response = client.post("/predict", files={
            "stream": buffer,
            "meta": json.dumps({"preprocessed": False})
            })
        assert response.status_code == 200
        # Preprocessed image
        response = client.post("/predict", files={
            "stream": binary_data,
            "meta": json.dumps({"preprocessed": True})
            })
        assert response.status_code == 200

@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="requires CUDA")
def test_torch_gpu():
    server = ls.LitServer(YOLOAI(enable_async=True), accelerator="cuda", devices=1)
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        # Unprocessed image
        response = client.post("/predict", files={
            "stream": buffer,
            "meta": json.dumps({"preprocessed": False})
            })
        assert response.status_code == 200
        # Preprocessed image
        response = client.post("/predict", files={
            "stream": binary_data,
            "meta": json.dumps({"preprocessed": True})
            })
        assert response.status_code == 200
