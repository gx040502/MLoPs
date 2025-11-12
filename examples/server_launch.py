import litserve as ls
from ModelManager.base import YOLOAI

if __name__ == "__main__":
    api = YOLOAI(
        model_path='./assests/best.pt',
        enable_async=True,
        # max_batch_size=8,
        # batch_timeout=0.1, 
        )
    server = ls.LitServer(api, accelerator="auto")
    server.run(host='0.0.0.0', port=8000, generate_client_file=False)
    # server.run(host='127.0.0.1', port=8000, generate_client_file=False)
