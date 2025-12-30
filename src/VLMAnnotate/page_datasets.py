import gradio as gr
from src.VLMAnnotate.ui_logic import refresh_all_components, cleanup_and_refresh_ui
from src.VLMAnnotate.tabs import upload_tab, vlm_tab, train_tab, predict_tab, pretrained_models_tab, about_tab

def load_dataset_interface(app_interface, app):
    # Custom CSS for the button
    gr.HTML("""
        <style>
        #btn {
            background: linear-gradient(45deg, #11998e, #38ef7d);
            border: none;
            color: white !important;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        #del_btn {
            background: linear-gradient(45deg, #FF416C, #FF4B2B);
            border: none;
            color: white !important;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        #del_btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        }
        #btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        }
        /* Global Slider Styling */
        input[type=range] {
            accent-color: #667eea !important; 
            filter: hue-rotate(240deg);
        }
        /* For Firefox */
        input[type=range]::-moz-range-thumb {
            background-color: #667eea !important;
        }
        </style>
    """)
    
    # --- Global State ---
    datasets_state = gr.State([])

    gr.Markdown(
        """
        <div class="page-header">
            <h1>🗂️ Data Management Systemm</h1>
            <p>Upload, manage, and organize your datasets</p>
        </div>
        """, 
        elem_classes=["page-header"]
    )
    
    with gr.Tabs() as tabs:
        # Create tabs
        upload_comps = upload_tab.create_tab(app)
        vlm_comps = vlm_tab.create_tab(app)
        train_comps = train_tab.create_tab(app)
        predict_comps = predict_tab.create_tab(app)
        pretrained_comps = pretrained_models_tab.create_tab(app)
        about_comps = about_tab.create_tab(app)
    
    # Registry of all components for cross-tab access
    all_components = {
        "datasets_state": datasets_state, 
        "tabs": tabs,                     
        "upload_tab": upload_comps,                 #ALL components from upload tab
        "vlm_tab": vlm_comps,
        "train_tab": train_comps,
        "predict_tab": predict_comps,
        "pretrained_models_tab": pretrained_comps,
        "about_tab": about_comps
    }
    
    # Setup events for each tab
    upload_tab.setup_events(app, upload_comps, all_components)
    vlm_tab.setup_events(app, vlm_comps, all_components)
    train_tab.setup_events(app, train_comps, all_components)
    predict_tab.setup_events(app, predict_comps, all_components)
    pretrained_models_tab.setup_events(app, pretrained_comps, all_components)
    about_tab.setup_events(app, about_comps, all_components)
    
    # --- Global Tab Event Handlers ---
    # Refresh all components when switching tabs
    # We pass necessary components to cleanup_and_refresh_ui via lambda or wrapper
    
    def on_tab_change(cvat_projects_val):
        return cleanup_and_refresh_ui(app, cvat_projects_val)

    common_outputs = [
        upload_comps["dataset_dropdown"], 
        vlm_comps["vlm_dataset_dropdown"], 
        train_comps["formatted_dataset_dropdown_train"], 
        train_comps["cvat_tasks_dropdown"]
    ]

    upload_comps["tab"].select(
        fn=on_tab_change,
        inputs=[train_comps["cvat_projects_dropdown"]], 
        outputs=common_outputs
    )
    # VLM tab select event removed - it was overriding the dataset selection from "Annotate" button
    # vlm_comps["tab"].select(
    #      fn=on_tab_change,
    #     inputs=[train_comps["cvat_projects_dropdown"]], 
    #     outputs=common_outputs
    # )
    train_comps["tab"].select(
        fn=on_tab_change,
        inputs=[train_comps["cvat_projects_dropdown"]], 
        outputs=common_outputs
    )
    predict_comps["tab"].select(
        fn=on_tab_change,
        inputs=[train_comps["cvat_projects_dropdown"]], 
        outputs=common_outputs
    )
    about_comps["tab"].select(
        fn=on_tab_change,
        inputs=[train_comps["cvat_projects_dropdown"]], 
        outputs=common_outputs
    )
    
    # --- App Initialization ---
    app_interface.load(
        fn=lambda x: refresh_all_components(app, x),
        inputs=[train_comps["cvat_projects_dropdown"]],
        outputs=[datasets_state, upload_comps["dataset_dropdown"], upload_comps["status_output"], train_comps["cvat_tasks_dropdown"]]
    )

if __name__ == "__main__":
    from app import APP
    app = APP()

    with gr.Blocks(title="See.AI Agent") as app_interface:
        load_dataset_interface(app_interface, app)
        gr.Button("Go to VLM", link="/vlm")

    app_interface.launch(
        debug=True,
        share=False,
        server_name="0.0.0.0",
        server_port=6605
    )
