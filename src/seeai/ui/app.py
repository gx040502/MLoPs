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
        
        /* Premium Login Page CSS */
        #login_container {
            max-width: 480px;
            margin: 10vh auto;
            padding: 40px;
            background: rgba(25, 25, 30, 0.6);
            backdrop-filter: blur(16px);
            border-radius: 24px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.08);
            text-align: center;
        }
        
        #login_title {
            font-size: 2.8em;
            font-weight: 900;
            background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
            font-family: 'Inter', sans-serif;
        }
        
        #login_subtitle {
            font-size: 1.05em;
            color: #aaa;
            margin-bottom: 30px;
            font-weight: 500;
        }
        
        #login_btn {
            background: linear-gradient(45deg, #4facfe, #00f2fe);
            border: none;
            color: white !important;
            font-weight: 700;
            border-radius: 12px;
            padding: 12px 20px;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 20px;
            font-size: 1.1em;
        }
        #login_btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(79, 172, 254, 0.5);
        }
        
        #gdrive_btn_auth {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: #ddd !important;
            border-radius: 12px !important;
            margin-top: 15px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        #gdrive_btn_auth:hover {
            background: rgba(255, 255, 255, 0.1) !important;
            border-color: #4facfe !important;
            color: #fff !important;
        }
        
        /* Remove Column gaps to not mess up main app */
        #main_app_container {
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
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

def build_ui(app_interface, app):
    with gr.Column(visible=True, elem_id="login_container") as login_page:
        gr.HTML("<div id='login_title'>SEE AI</div>")
        gr.HTML("<div id='login_subtitle'>Before we begin, do you have a CVAT account?</div>")
        
        has_cvat = gr.Radio(["Yes", "No"], label="", value=None, container=False)
        
        with gr.Column(visible=False, variant="panel") as cvat_creds_group:
            gr.Markdown("### 🔑 CVAT Credentials")
            cvat_user = gr.Textbox(label="Username", placeholder="Enter your CVAT username")
            cvat_pass = gr.Textbox(label="Password", type="password", placeholder="Enter your CVAT password")
            
            gr.HTML("<hr style='border-color: rgba(255,255,255,0.1); margin: 20px 0;'>")
            gr.Markdown("### ☁️ Google Drive (Optional)")
            gr.Markdown("<span style='color: #888; font-size: 0.9em;'>Automatically export your trained weights to the cloud.</span>")
            gdrive_btn = gr.Button("🔗 Authenticate via Google", elem_id="gdrive_btn_auth")
            gdrive_status = gr.Markdown("<center><span style='color: #888; font-size: 0.8em;'>Not connected</span></center>")
            
        enter_btn = gr.Button("Enter Workspace 🚀", elem_id="login_btn")
            
    with gr.Column(visible=False, elem_id="main_app_container") as main_app:
        all_comps = load_dataset_interface(app_interface, app)
        
    def toggle_creds(val):
        return gr.update(visible=(val == "Yes"))
        
    def handle_gdrive_auth():
        success, msg = authenticate_gdrive()
        if success:
            return f"<center><span style='color: #38ef7d; font-size: 0.8em;'>✅ {msg}</span></center>"
        return f"<center><span style='color: #FF416C; font-size: 0.8em;'>❌ {msg}</span></center>"
        
    def handle_login(has_cvat_val, user, pwd):
        if has_cvat_val == "No" or not has_cvat_val:
            # User has no CVAT, so we MUST hide all CVAT-dependent tabs.
            # In Gradio, we update the visibility of the specific Tabs.
            return [
                gr.update(visible=False), # Hide login page
                gr.update(visible=True),  # Show main app container
                gr.update(visible=True),  # Auto Annotation (VLM)
                gr.update(visible=False), # Train
                gr.update(visible=False), # Predict
                gr.update(visible=False), # Pretrained
                gr.update(visible=False)  # About
            ]
        else:
            # User HAS CVAT account, save creds to env and allow access to all tabs
            env_path = "src/seeai/config/.env"
            if os.path.exists(env_path):
                dotenv.set_key(env_path, "CVAT_USER", user)
                dotenv.set_key(env_path, "CVAT_PASSWORD", pwd)
                os.environ["CVAT_USER"] = user
                os.environ["CVAT_PASSWORD"] = pwd
                
            return [
                gr.update(visible=False), # Hide login page
                gr.update(visible=True),  # Show main app container
                gr.update(visible=True),  # Auto Annotation (VLM)
                gr.update(visible=True),  # Train
                gr.update(visible=True),  # Predict
                gr.update(visible=True),  # Pretrained
                gr.update(visible=True)   # About
            ]
            
    has_cvat.change(fn=toggle_creds, inputs=[has_cvat], outputs=[cvat_creds_group])
    gdrive_btn.click(fn=handle_gdrive_auth, outputs=[gdrive_status])
    
    # We output to the exact tab components to control their visibility
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

if __name__ == "__main__":
    from src.seeai.core.annotation_engine import APP
    app = APP()

    with gr.Blocks(title="See.AI Agent") as app_interface:
        build_ui(app_interface, app)

    app_interface.launch(
        debug=True,
        share=False,
        server_name="0.0.0.0",
        server_port=6605
    )
