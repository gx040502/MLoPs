import base64
import pytest

import torch
import litserve as ls

from fastapi.testclient import TestClient
from litserve.utils import wrap_litserve_start

from ModelManager.base import YOLOAI

sample = './assests/zapfan.jpeg'
data = base64.b64encode(open(sample, 'rb').read()).decode('utf-8')

def test_torch_cpu():
    server = ls.LitServer(YOLOAI(), accelerator="cpu", devices=1)
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"image_data": data})
        assert response.status_code == 200

@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="requires CUDA")
def test_torch_gpu():
    server = ls.LitServer(YOLOAI(), accelerator="cuda", devices=1)
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"image_data": data})
        assert response.status_code == 200
