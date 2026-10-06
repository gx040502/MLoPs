import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from src.seeai.utils.gdrive_auth import authenticate_gdrive
from google.oauth2.credentials import Credentials

def get_drive_service():
    """Builds and returns the Google Drive API service."""
    token_path = "token.json"
    if not os.path.exists(token_path):
        success, _ = authenticate_gdrive()
        if not success:
            return None
    creds = Credentials.from_authorized_user_file(token_path, ['https://www.googleapis.com/auth/drive.file'])
    return build('drive', 'v3', credentials=creds)

def upload_file_to_gdrive(local_path, filename, folder_id=None):
    """Uploads a single file to Google Drive."""
    service = get_drive_service()
    if not service: return None
    
    file_metadata = {'name': filename}
    if folder_id:
        file_metadata['parents'] = [folder_id]
        
    media = MediaFileUpload(local_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')

def upload_folder_to_gdrive(local_folder_path, gdrive_folder_name):
    """Uploads an entire directory of trained model outputs to Google Drive."""
    service = get_drive_service()
    if not service: return None
    
    # 1. Create a parent folder in Drive
    folder_metadata = {
        'name': gdrive_folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    parent_folder = service.files().create(body=folder_metadata, fields='id').execute()
    parent_id = parent_folder.get('id')
    
    # 2. Upload all files recursively (simplified, assuming flat or few levels)
    # We really just need to upload weights/best.pt and results.png, etc.
    for root, _, files in os.walk(local_folder_path):
        for file in files:
            local_path = os.path.join(root, file)
            # Create a subfolder if needed, but for simplicity let's just upload best.pt and results
            if file == "best.pt" or file.endswith(('.png', '.jpg')):
                upload_file_to_gdrive(local_path, file, parent_id)
                
    return parent_id
