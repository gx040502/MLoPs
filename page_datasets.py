import gradio as gr
import os
import json
import random
import time
from typing import List, Dict, Any


def load_dataset_interface(app_interface, app):
    """Load the dataset management interface"""
    
    def refresh_datasets_state():
        """Get current datasets and return as state"""
        return app.get_all_datasets()

    def select_dataset(dataset_name: str) -> str:
        success, message = app.select_dataset(dataset_name)
        return message

    def delete_dataset_handler(dataset_name: str) -> tuple:
        """Handle dataset deletion and return updated state"""
        result = app.remove_dataset(dataset_name)
        updated_datasets = refresh_datasets_state()
        datasets_html = app.create_dataset_html()
        return result, updated_datasets, datasets_html

    def upload_and_refresh(zip_file) -> tuple:
        """Handle upload and refresh datasets"""
        upload_result = app.upload_dataset_by_zip(zip_file)
        updated_datasets = refresh_datasets_state()
        return upload_result, updated_datasets

    # Global state to store datasets
    datasets_state = gr.State(value=refresh_datasets_state())
    
    gr.Markdown(
        """
        <div class="page-header">
            <h1>🗂️ Data Management System</h1>
            <p>Upload, manage, and organize your datasets</p>
        </div>
        """, 
        elem_classes=["page-header"]
    )
    
    with gr.Tabs() as tabs:
        
        # HOME/DATASETS TAB
        with gr.Tab("📊 Home - Datasets"):
            with gr.Column():
                gr.Markdown("## Available Datasets")
            
                # Datasets display
                datasets_display = gr.HTML(
                    value=app.create_dataset_html(),
                    label="Datasets"
                )
                
                # Status display
                status_output = gr.Textbox(
                    label="Selected Dataset", 
                    interactive=False, 
                    lines=2,
                    value="Ready"
                )

                # Dataset selection
                with gr.Row():
                    dataset_dropdown = gr.Dropdown(
                        choices=[d["name"] for d in refresh_datasets_state()],
                        label="Select Dataset",
                        interactive=True
                    )
                    with gr.Column():
                        annotate_btn = gr.Button("📂 Annotate", variant="primary")
                        delete_btn = gr.Button("🗑️ Delete Selected Dataset", variant="secondary")
                

        # UPLOAD TAB
        with gr.Tab("📤 Upload Dataset"):
            gr.Markdown("## Upload New Dataset")
            gr.Markdown("Upload a ZIP file containing your dataset. It will be extracted and added to the available datasets.")
            
            with gr.Column():
                zip_file_input = gr.File(
                    label="Upload ZIP File", 
                    file_count="single", 
                    file_types=[".zip"], 
                    type="filepath"
                )
                
                upload_btn = gr.Button("📤 Upload & Extract", variant="primary", size="lg")
                
                upload_status = gr.Textbox(
                    label="Upload Status", 
                    interactive=False, 
                    lines=10,
                    value="Ready to upload..."
                )
        
        # ABOUT TAB
        with gr.Tab("ℹ️ About"):
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
            3. Select datasets from the dropdown to view details\
            
            ### File Structure:
            - Uploaded datasets are extracted to `unzipped_files/`
            - Dataset metadata is stored in `settings.json`
            - Each dataset entry includes name, path, and upload timestamp
            """)
    
    # Event handlers
    def update_dataset_dropdown_and_display(datasets):
        """Update both dropdown choices and HTML display"""
        choices = [d["name"] for d in datasets]
        html = app.create_dataset_html()
        return gr.update(choices=choices, value=None), html
    
    def refresh_all_components():
        """Refresh all dataset-related components"""
        updated_datasets = refresh_datasets_state()
        dropdown_update, html_update = update_dataset_dropdown_and_display(updated_datasets)
        return updated_datasets, dropdown_update, html_update, "Datasets refreshed!"
    
    # When selected dataset changes
    dataset_dropdown.change(
        fn=lambda dataset_name: app.select_dataset(dataset_name) if dataset_name else "Please select a dataset first",
        inputs=[dataset_dropdown],
        outputs=status_output
    )

    annotate_btn.click(
        None, None, None, js="() => window.location.href = '/vlm'"  # Redirect to VLM tab after processing
    )
    
    delete_btn.click(
        fn=lambda dataset_name, datasets: (
            delete_dataset_handler(dataset_name) if dataset_name 
            else ("Please select a dataset first", datasets, app.create_dataset_html())
        ),
        inputs=[dataset_dropdown, datasets_state],
        outputs=[status_output, datasets_state, datasets_display]
    ).then(
        fn=update_dataset_dropdown_and_display,
        inputs=datasets_state,
        outputs=[dataset_dropdown, datasets_display]
    )
    
    upload_btn.click(
        fn=upload_and_refresh,
        inputs=zip_file_input,
        outputs=[upload_status, datasets_state]
    ).then(
        fn=update_dataset_dropdown_and_display,
        inputs=datasets_state,
        outputs=[dataset_dropdown, datasets_display]
    )
    
    # Initialize the interface
    app_interface.load(
        fn=refresh_all_components,
        outputs=[datasets_state, dataset_dropdown, datasets_display, status_output]
    )

if __name__ == "__main__":

    from app import APP
    app = APP()

    with gr.Blocks(title="See.AI Agent") as app_interface:
        load_dataset_interface(app)
        gr.Button("Go to VLM", link="/vlm")

    app_interface.launch(
        debug=True,
        share=False,
        server_name="0.0.0.0",
        server_port=6605
    )