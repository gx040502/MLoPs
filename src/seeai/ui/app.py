import os
import gradio as gr
from src.seeai.ui.ui_logic import refresh_all_components
from src.seeai.ui.tabs import vlm_tab, train_tab, predict_tab, pretrained_tab, about_tab
from src.seeai.utils.gdrive_auth import authenticate_gdrive
import dotenv

def load_dataset_interface(app_interface, app):
    # Custom CSS for the button
    gr.HTML("""
        <style>
        .page-header {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 5px;
        }
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
            <h1>🗂️ SEE AI</h1>
        </div>
        """
    )
    
    with gr.Tabs() as tabs:
        vlm_comps = vlm_tab.create_tab(app)
        train_comps = train_tab.create_tab(app)
        predict_comps = predict_tab.create_tab(app)
        pretrained_comps = pretrained_tab.create_tab(app)
        about_comps = about_tab.create_tab(app)
    
    # Registry of all components for cross-tab access
    all_components = {
        "datasets_state": datasets_state, 
        "tabs": tabs,                     
        "vlm_tab": vlm_comps,
        "train_tab": train_comps,
        "predict_tab": predict_comps,
        "pretrained_tab": pretrained_comps,
        "about_tab": about_comps
    }
    
    # Setup events for each tab
    vlm_tab.setup_events(app, vlm_comps, all_components)
    train_tab.setup_events(app, train_comps, all_components)
    predict_tab.setup_events(app, predict_comps, all_components)
    pretrained_tab.setup_events(app, pretrained_comps, all_components)
    about_tab.setup_events(app, about_comps, all_components)
    
    # --- App Initialization ---
    app_interface.load(
        fn=lambda x: refresh_all_components(app, x),
        inputs=[train_comps["cvat_projects_dropdown"]],
        outputs=[datasets_state]
    )

    return all_components


if __name__ == "__main__":
    from src.seeai.core.annotation_engine import APP
    app = APP()

    with gr.Blocks(title="See.AI Agent") as app_interface:
        
        with gr.Group(visible=True) as login_page:
            gr.Markdown("<h1 style='text-align: center; margin-top: 50px;'>Welcome to SEE AI</h1>")
            gr.Markdown("<h3 style='text-align: center;'>Before we begin, do you have a CVAT account?</h3>")
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    pass
                with gr.Column(scale=2):
                    has_cvat = gr.Radio(["Yes", "No"], label="", value=None)
                    
                    with gr.Group(visible=False) as cvat_creds_group:
                        gr.Markdown("### 🔑 Enter CVAT Credentials")
                        cvat_user = gr.Textbox(label="CVAT Username")
                        cvat_pass = gr.Textbox(label="CVAT Password", type="password")
                        
                        gr.Markdown("### ☁️ Google Drive Integration (Optional)")
                        gr.Markdown("Save training outputs directly to Google Drive via Google API.")
                        gdrive_btn = gr.Button("Authenticate Google Drive")
                        gdrive_status = gr.Markdown("*Not connected*")
                        
                    enter_btn = gr.Button("Enter System", variant="primary", size="lg")
                with gr.Column(scale=1):
                    pass
                    
        with gr.Group(visible=False) as main_app:
            all_comps = load_dataset_interface(app_interface, app)
            
        def toggle_creds(val):
            return gr.update(visible=(val == "Yes"))
            
        def handle_gdrive_auth():
            success, msg = authenticate_gdrive()
            if success:
                # Tell the .env to use a special flag or we just know it's auth'd.
                # Actually, our model_trainer.py looks for GDRIVE_OUTPUT_DIR path.
                # If they use OAuth, the files aren't saved to a local path, they are uploaded via API!
                # Wait, earlier we set GDRIVE_OUTPUT_DIR="G:\My Drive...".
                # If they authenticate via API, they want cloud upload, not local save.
                return f"✅ {msg}"
            return f"❌ {msg}"
            
        def handle_login(has_cvat_val, user, pwd):
            if has_cvat_val == "No" or not has_cvat_val:
                # Hide training and predict tabs, show only auto annotation
                return [
                    gr.update(visible=False), # login
                    gr.update(visible=True),  # main
                    gr.update(visible=True),  # vlm tab
                    gr.update(visible=False), # train tab
                    gr.update(visible=False), # predict tab
                    gr.update(visible=False), # pretrained tab
                    gr.update(visible=False)  # about tab
                ]
            else:
                # Save to .env
                env_path = "src/seeai/config/.env"
                if os.path.exists(env_path):
                    dotenv.set_key(env_path, "CVAT_USER", user)
                    dotenv.set_key(env_path, "CVAT_PASSWORD", pwd)
                    # Force reload inside os.environ for current session
                    os.environ["CVAT_USER"] = user
                    os.environ["CVAT_PASSWORD"] = pwd
                    
                return [
                    gr.update(visible=False), # login
                    gr.update(visible=True),  # main
                    gr.update(visible=True),  # vlm tab
                    gr.update(visible=True),  # train tab
                    gr.update(visible=True),  # predict tab
                    gr.update(visible=True),  # pretrained tab
                    gr.update(visible=True)   # about tab
                ]
                
        has_cvat.change(fn=toggle_creds, inputs=[has_cvat], outputs=[cvat_creds_group])
        gdrive_btn.click(fn=handle_gdrive_auth, outputs=[gdrive_status])
        
        enter_btn.click(
            fn=handle_login,
            inputs=[has_cvat, cvat_user, cvat_pass],
            outputs=[
                login_page, 
                main_app,
                all_comps["vlm_tab"]["tab"],
                all_comps["train_tab"]["tab"],
                all_comps["predict_tab"]["tab"],
                all_comps["pretrained_tab"]["tab"],
                all_comps["about_tab"]["tab"]
            ]
        )

    app_interface.launch(
        debug=True,
        share=False,
        server_name="0.0.0.0",
        server_port=6605
    )
