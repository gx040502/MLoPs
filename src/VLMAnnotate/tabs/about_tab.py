import gradio as gr

def create_tab(app):
    with gr.Tab("ℹ️ About") as tab:
        gr.Markdown("""
        ## About Dataset Manager
        
        This application helps you manage your datasets efficiently:
        
        ### Features:
        - 📤 **Upload**: Upload ZIP files containing your datasets
        - 📊 **View**: Browse all available datasets with details
        - 🔄 **Refresh**: Real-time updates when datasets change
        - 🗑️ **Delete**: Remove datasets you no longer need
        - 💾 **Persist**: All dataset information is saved in settings.json
        
        ### How to use:
        1. Go to the **Upload** tab to add new datasets
        2. Use the **Home** tab to view and manage existing datasets
        3. Select datasets from the dropdown to view details
        
        ### File Structure:
        - Uploaded datasets are extracted to `unzipped_files/`
        - Dataset metadata is stored in `settings.json`
        - Each dataset entry includes name, path, and upload timestamp
        """)
        
    return {"tab": tab}

def setup_events(app, components, all_components):
    c = components
    c["tab"].select(fn=lambda _: app.cleanup_preview(), outputs=None)
