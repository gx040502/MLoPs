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
        
        #gdrive_btn_auth {
            background: transparent !important;
            border: 1px solid #4facfe !important;
            color: #4facfe !important;
            font-weight: bold;
        }
        #gdrive_btn_auth:hover {
            background: rgba(79, 172, 254, 0.1) !important;
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
        
        # NEW: Setup & Login Tab integrated directly into the layout
        with gr.Tab("🔑 Setup & Login", id="setup_tab"):
            gr.Markdown("## Welcome to SEE AI")
            gr.Markdown("Auto Annotation (VLM), Pretrained Models, and About sections are completely free to use without an account.\nHowever, to Train models and Predict on CVAT projects, you must connect your CVAT account.")
            
            with gr.Row():
                with gr.Column(variant="panel"):
                    gr.Markdown("### 🔑 Connect CVAT Account")
                    cvat_user = gr.Textbox(label="Username", placeholder="Enter your CVAT username")
                    cvat_pass = gr.Textbox(label="Password", type="password", placeholder="Enter your CVAT password")
                    login_btn = gr.Button("Authenticate CVAT", variant="primary")
                    login_status = gr.Markdown("<span style='color: #888;'>*Not logged in*</span>")
                
                with gr.Column(variant="panel"):
                    gr.Markdown("### ☁️ Google Drive (Optional)")
                    gr.Markdown("Automatically export your trained weights to the cloud.")
                    gdrive_btn = gr.Button("🔗 Authenticate via Google", elem_id="gdrive_btn_auth")
                    gdrive_status = gr.Markdown("<span style='color: #888;'>*Not connected*</span>")
                    
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
    
    # --- Setup & Login Logic ---
    def handle_cvat_login(user, pwd):
        if not user or not pwd:
            return [
                "<span style='color: #FF416C;'>❌ Please enter both username and password</span>",
                gr.update(),
                gr.update()
            ]
            
        try:
            from cvat_sdk import make_client
            from src.seeai.config.settings import CVAT_HOST_IP, CVAT_HOST_PORT
            
            # Sanitize URL just in case
            url = f"{CVAT_HOST_IP}:{CVAT_HOST_PORT}"
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
                
            # Validating credentials by attempting to connect
            with make_client(url, credentials=(user, pwd)) as client:
                pass # If it doesn't throw, credentials are valid!
        except Exception as e:
            return [
                "<span style='color: #FF416C;'>❌ Authentication Failed: Invalid credentials or server offline</span>",
                gr.update(),
                gr.update()
            ]
            
        # Save creds to env and allow access to locked tabs
        env_path = "src/seeai/config/.env"
        if os.path.exists(env_path):
            dotenv.set_key(env_path, "CVAT_USER", user)
            dotenv.set_key(env_path, "CVAT_PASSWORD", pwd)
            os.environ["CVAT_USER"] = user
            os.environ["CVAT_PASSWORD"] = pwd
            
        return [
            "<span style='color: #38ef7d;'>✅ Successfully Authenticated! Tabs Unlocked.</span>",
            gr.update(interactive=True, label="🧠 Train Model"),
            gr.update(interactive=True, label="🎱 Predict Model")
        ]

    def handle_gdrive_auth():
        success, msg = authenticate_gdrive()
        if success:
            return f"<span style='color: #38ef7d;'>✅ {msg}</span>"
        return f"<span style='color: #FF416C;'>❌ {msg}</span>"
        
    login_btn.click(
        fn=handle_cvat_login,
        inputs=[cvat_user, cvat_pass],
        outputs=[login_status, all_components["train_tab"]["tab"], all_components["predict_tab"]["tab"]]
    )
    
    gdrive_btn.click(fn=handle_gdrive_auth, outputs=[gdrive_status])
    
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
        load_dataset_interface(app_interface, app)

    app_interface.launch(
        debug=True,
        share=False,
        server_name="0.0.0.0",
        server_port=6605
    )
