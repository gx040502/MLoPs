import os
import dotenv

from cvat_sdk import Client

dotenv.load_dotenv()

CVAT_HOST_IP = os.getenv("CVAT_HOST_IP")
CVAT_HOST_PORT = os.getenv("CVAT_HOST_PORT")
CVAT_USER = os.getenv("CVAT_USER")
CVAT_PASSWORD = os.getenv("CVAT_PASSWORD")
GDRIVE_OUTPUT_DIR = os.getenv("GDRIVE_OUTPUT_DIR")

