import os
import dotenv

from cvat_sdk import Client

dotenv.load_dotenv()

CVAT_URL = os.getenv("CVAT_URL", "http://localhost:8080")
CVAT_USER = os.getenv("CVAT_USER")
CVAT_PASSWORD = os.getenv("CVAT_PASSWORD")
GDRIVE_OUTPUT_DIR = os.getenv("GDRIVE_OUTPUT_DIR")

