import gradio as gr

def create_tab(app):
    with gr.Tab("🧠 Train Model", id="train_tab") as tab:
        gr.Markdown("## Train New Model")
        gr.Markdown("Train a new model from your CVAT Project")
        
        with gr.Column():
            # Get initial CVAT projects and select first one
            cvat_projects_initial = app.get_cvat_projects()
            cvat_initial_value = cvat_projects_initial[0][1] if cvat_projects_initial else None
            
            cvat_projects_dropdown = gr.Dropdown(
                label="Choose CVAT Project", 
                choices=cvat_projects_initial,
                value=cvat_initial_value,
                visible=True, 
                interactive=True
            )
            with gr.Row():
                with gr.Column():
                    format_dropdown = gr.Dropdown(
                        label="Choose Format",
                        choices=["Ultralytics YOLO Detection 1.0", "Ultralytics YOLO Segmentation 1.0","Ultralytics YOLO Classification 1.0"],
                        value="Ultralytics YOLO Detection 1.0",
                        interactive=True
                    )
                    
                    model_dropdown = gr.Dropdown(
                            choices=app.get_pretrained_models("Ultralytics YOLO Detection 1.0"),
                            label="Select Pre-trained Model",
                            interactive=True
                        )
                    
                with gr.Column():
                        model_name_input = gr.Textbox(
                            label="Model Name (Optional)",
                            placeholder="Leave empty for auto-generated name",
                            value=""
                        )
                        model_version_input = gr.Textbox(
                            label="Version (Optional)",
                            placeholder="Leave empty for auto-increment (v1, v2, ...)",
                            value=""
                        )
                
            train_config_checkbox = gr.Checkbox(
                value=False,
                label="⚙️ Set Own Training Configuration",
                elem_classes="train-config-checkbox"
            )

            # Custom styling for Training Configuration section
            gr.HTML("""
                <style>
                    /* Control checkbox spacing */
                    .train-config-checkbox {
                        margin-bottom: 0px !important;
                    }
                    /* Remove padding from HTML container wrapper */
                    .html-container.svelte-phx28p {
                        padding: 0 !important;
                        margin: 0 !important;
                    }
                    /* Remove spacing from block containers */
                    .block.svelte-1svsvh2 {
                        margin-top: 0 !important;
                        margin-bottom: 0 !important;
                    }
                    .training-config-container {
                        background: linear-gradient(135deg, #1e3a8a10 0%, #3b82f615 100%) !important;
                        border: 2px solid #3b82f6 !important;
                        border-radius: 12px !important;
                        padding: 10px !important;
                        margin: 0 !important;
                        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.1) !important;
                    }
                    /* Override Gradio's default group padding */
                    .training-config-container > .form {
                        padding: 0 !important;
                        gap: 10px !important;
                    }
                    /* Remove spacing from parent containers */
                    .training-config-container {
                        margin-top: 0 !important;
                        margin-bottom: 0 !important;
                    }
                    .training-config-header {
                        background: linear-gradient(90deg, #3b82f6, #6366f1);
                        color: white;
                        padding: 12px;
                        border-radius: 12px;
                        margin-bottom: 10px;
                        font-weight: 600;
                        font-size: 1.1em;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                </style>
            """)

            with gr.Group(visible=False, elem_classes="training-config-container") as train_config_group:
                gr.HTML('<div class="training-config-header">⚙️ Training Configuration</div>')
                with gr.Row():
                    experiments_slider = gr.Slider(
                        minimum=1, 
                        maximum=1000, 
                        value=100, 
                        step=1, 
                        label="Epochs"
                    )
                    
                    imgsz_slider = gr.Slider(
                        minimum=64, 
                        maximum=1280, 
                        value=640, 
                        step=32, 
                        label="Image Size"
                    )

                
                manual_aug_checkbox = gr.Checkbox(
                    value=False,
                    label="Manual Adjust Augmentation"
                )
                    
                base_image_state = gr.State(None)
                    
                with gr.Group(visible=False) as aug_settings_group:
                    with gr.Row():
                        with gr.Column():
                            aug_preview_img = gr.Image(label="Augmentation Preview", interactive=False, type="pil")
                            gr.Markdown("### 👁️ Live Preview\nClick below to load a random image from the task to see how augmentations affect it.")
                            load_sample_btn = gr.Button("🎲 Load New Sample", size="sm")

                        with gr.Column():
                            gr.Markdown("#### 🎨 Color Augmentation")
                            with gr.Row():
                                hsv_h = gr.Slider(0.0, 1.0, value=0.015, step=0.001, label="HSV-Hue")
                                hsv_s = gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="HSV-Saturation")
                            with gr.Row():
                                hsv_v = gr.Slider(0.0, 1.0, value=0.4, step=0.01, label="HSV-Value")
                                bgr = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="BGR Flip Prob")

                            gr.Markdown("#### 📐 Geometric Transforms")
                            with gr.Row():
                                degrees = gr.Slider(-180, 180, value=0.0, step=1.0, label="Rotation (+/- deg)")
                                translate = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="Translate (+/- frac)")
                            with gr.Row():
                                scale = gr.Slider(0.0, 2.0, value=0.5, step=0.01, label="Scale (+/- gain)")
                                shear = gr.Slider(-180, 180, value=0.0, step=1.0, label="Shear (+/- deg)")
                            with gr.Row():
                                perspective = gr.Slider(0.0, 0.001, value=0.0, step=0.0001, label="Perspective")
                                flipud = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Flip Up-Down Prob")
                            with gr.Row():
                                fliplr = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Flip Left-Right Prob")

                            gr.Markdown("#### 🔀 Advanced Mixing")
                            with gr.Row():
                                mosaic = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Mosaic Prob")
                                mixup = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Mixup Prob")
                            with gr.Row():
                                cutmix = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Cutmix Prob")
                                copy_paste = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Copy-Paste Prob")

                            gr.Markdown("#### ✨ Image Quality & Effects")
                            with gr.Row():
                                erasing = gr.Slider(0.0, 0.9, value=0.4, step=0.01, label="Erasing %")

            train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg", elem_id="btn")
            gr.HTML("""
                <div style="text-align: center; margin: 15px 0;">
                    <div style="
                        height: 1px; 
                        width: 100%;
                        max-width: 100%;
                        margin: 0 auto;
                        background: linear-gradient(90deg, transparent, #6366f1, transparent); 
                        box-shadow: 0 0 10px rgba(99, 102, 241, 0.5);">
                    </div>
                </div>
            """)
            
            dataset_log = gr.Textbox(
                label="Dataset Detail Log", 
                interactive=False, 
                lines=5,
                value="Waiting for dataset..."
            )
            
            training_log = gr.Textbox(
                label="Training Log", 
                interactive=False, 
                lines=2,
                value="Waiting to start..."
            )
            # ETA Components
            eta_output = gr.Markdown("⏳ Estimated Time: Waiting to start...", visible=False)
            current_training_project_name = gr.State(None) 
            
            filename_input = gr.Textbox(
                label="Custom Output Filename (Optional but this will be the train model name)", 
                placeholder="e.g. my_dataset.zip",
                lines=1,
                visible=False
            )

            gr.HTML("<div style='margin-top: 5px;'></div>")

    return {
        "tab": tab,
        "cvat_projects_dropdown": cvat_projects_dropdown,
        "format_dropdown": format_dropdown,
        "model_dropdown": model_dropdown,
        "train_config_checkbox": train_config_checkbox,
        "train_config_group": train_config_group,
        "experiments_slider": experiments_slider,
        "imgsz_slider": imgsz_slider,
        "model_name_input": model_name_input,
        "model_version_input": model_version_input,
        "manual_aug_checkbox": manual_aug_checkbox,
        "aug_settings_group": aug_settings_group,
        "base_image_state": base_image_state,
        "aug_preview_img": aug_preview_img,
        "load_sample_btn": load_sample_btn,
        "hsv_h": hsv_h, "hsv_s": hsv_s, "hsv_v": hsv_v, "bgr": bgr,
        "degrees": degrees, "translate": translate, "scale": scale, "shear": shear,
        "perspective": perspective, "flipud": flipud, "fliplr": fliplr,
        "mosaic": mosaic, "mixup": mixup, "cutmix": cutmix, "copy_paste": copy_paste,
        "erasing": erasing,
        "train_btn": train_btn,
        "dataset_log": dataset_log,
        "training_log": training_log,
        "eta_output": eta_output,
        "current_training_project_name": current_training_project_name,
        "filename_input": filename_input
    }

def setup_events(app, components, all_components):
    c = components
    
    # Internal logic
    def on_format_change(format_val):   
        models = app.get_pretrained_models(format_val)
        return gr.update(choices=models, value=models[0] if models else None)
        
    def toggle_aug_settings(checkbox_val): 
        return gr.update(visible=checkbox_val)
                
    def trigger_training(project_id, format_name, custom_name, model, epochs, imgsz, model_name, model_version, manual_aug, 
                                     h_h, h_s, h_v, bgr_p, 
                                     deg, trans, scl, shr, 
                                     persp, f_ud, f_lr, 
                                     mos, mix, cut, cp, 
                                     ers): 
        # 1. Format/Prepare Data
        msg, path = app.process_cvat_project(project_id, custom_name, format_name=format_name)
        if "Error" in msg:
            yield msg, "❌ Format Failed", None
            return

        # Inspect Dataset Stats
        stats = app.inspect_dataset_zip(path)
        
        stats_str = "📊 Dataset Stats:\n"
        if stats.get("status") != "Error":
            stats_str += f"Valid Images: {stats['images']}\n"
            stats_str += f"Classes ({stats['classes']}): {stats['class_names']}\n"
        else:
            stats_str += f"Error inspecting stats: {stats.get('message')}\n"
                        
        yield f"{msg}\n\n🚀 Training Started...", stats_str, None

        # 2. Augmentation Params
        aug_args = {
            'hsv_h': h_h, 'hsv_s': h_s, 'hsv_v': h_v, 'bgr': bgr_p,
            'degrees': deg, 'translate': trans, 'scale': scl, 'shear': shr,
            'perspective': persp, 'flipud': f_ud, 'fliplr': f_lr,
            'mosaic': mos, 'mixup': mix, 'cutmix': cut, 'copy_paste': cp,
            'erasing': ers
        }
                    
        # 3. Start Training
        project_name = ""
        if custom_name and custom_name.strip():
            s_name = custom_name.strip()
            if s_name.lower().endswith('.zip'): 
                project_name = s_name[:-4]
            else:
                project_name = s_name
        else:
            clean_id = str(project_id).split(':')[0].strip()
            project_name = f"Project_{clean_id}"
                        
        yield "🚀 Training Started... ETA should appear shortly.", stats_str, project_name
                    
        result = app.start_training(path, model, epochs, imgsz, manual_aug, project_id, model_name, model_version, format_name=format_name, **aug_args)
        yield result, stats_str, None
        
    def check_training_status(project_name): 
        if not project_name: return gr.update(visible=False)
                    
        from pathlib import Path
        import json

        # Direct path to the status file
        status_path = Path("Dataset") / project_name / "training_status.json"
                    
        if not status_path.exists():
            return gr.update(value=f"⏳ Estimated Time: Initializing... (Waiting for {project_name})", visible=True)
                        
        try:
            with open(status_path, 'r') as f:
                data = json.load(f)
                            
            rem_seconds = data.get("est_time_remaining", 0)
            
            def fmt(s):
                s = int(s)
                m, s = divmod(s, 60)
                return f"{m}m {s}s"
                            
            return gr.update(value=f"⏳ **Epoch {data['current_epoch']}/{data['total_epochs']}** | Remaining: {fmt(rem_seconds)}", visible=True)
        except Exception as e:
            return gr.update(value=f"⏳ Estimated Time: Error reading status ({e})", visible=True)

    def on_train_config_toggle(checked):
        """Toggle training config visibility"""
        return gr.update(visible=checked)

    def load_sample_image(project_str): 
        if not project_str: return None, None
        project_str = str(project_str) # Ensure string
        project_id = project_str.split(':')[0].strip() if ':' in project_str else project_str
        images = app.get_random_sample_images(project_id)
        if not images: return None, None
        return images, images[0] # State stores list, UI shows 1st image initially

    def update_aug_preview(images, h_h, h_s, h_v, bgr_p, deg, trans, scl, shr, persp, f_ud, f_lr, mos, mix, cut, cp, ers): 
        if not images: return None
        aug_args = {
            'hsv_h': h_h, 'hsv_s': h_s, 'hsv_v': h_v, 'bgr': bgr_p,
            'degrees': deg, 'translate': trans, 'scale': scl, 'shear': shr,
            'perspective': persp, 'flipud': f_ud, 'fliplr': f_lr,
            'mosaic': mos, 'mixup': mix, 'cutmix': cut, 'copy_paste': cp,
            'erasing': ers
        }
        return app.preview_augmentation(images, **aug_args)

    def refresh_cvat_projects():
        """Refresh CVAT projects dropdown when tab is selected"""
        projects = app.get_cvat_projects()
        current_value = projects[0][1] if projects else None
        return gr.update(choices=projects, value=current_value)
    
    # --- Event Handlers ---
    c["tab"].select(
        fn=refresh_cvat_projects,
        outputs=[c["cvat_projects_dropdown"]]
    )
    
    c["format_dropdown"].change(
        fn=on_format_change,
        inputs=[c["format_dropdown"]],
        outputs=[c["model_dropdown"]] 
    )
    
    c["manual_aug_checkbox"].change(fn=toggle_aug_settings, inputs=c["manual_aug_checkbox"], outputs=c["aug_settings_group"])

    eta_timer = gr.Timer(2)
    eta_timer.tick(
        fn=check_training_status, 
        inputs=[c["current_training_project_name"]], 
        outputs=[c["eta_output"]]
    )
    
    c["train_btn"].click(
        fn=trigger_training,
        inputs=[
            c["cvat_projects_dropdown"], c["format_dropdown"], c["filename_input"], c["model_dropdown"], c["experiments_slider"], c["imgsz_slider"], c["model_name_input"], c["model_version_input"], c["manual_aug_checkbox"],
            c["hsv_h"], c["hsv_s"], c["hsv_v"], c["bgr"],
            c["degrees"], c["translate"], c["scale"], c["shear"],
            c["perspective"], c["flipud"], c["fliplr"],
            c["mosaic"], c["mixup"], c["cutmix"], c["copy_paste"],
            c["erasing"]
        ],
        outputs=[c["training_log"], c["dataset_log"], c["current_training_project_name"]]
    )
    
    c["train_config_checkbox"].change(
        fn=on_train_config_toggle,
        inputs=[c["train_config_checkbox"]],
        outputs=[c["train_config_group"]]
    )
    
    c["load_sample_btn"].click(
        fn=load_sample_image,
        inputs=[c["cvat_projects_dropdown"]],
        outputs=[c["base_image_state"], c["aug_preview_img"]]
    )
    
    aug_inputs = [
        c["base_image_state"],
        c["hsv_h"], c["hsv_s"], c["hsv_v"], c["bgr"],
        c["degrees"], c["translate"], c["scale"], c["shear"],
        c["perspective"], c["flipud"], c["fliplr"],
        c["mosaic"], c["mixup"], c["cutmix"], c["copy_paste"],
        c["erasing"]
    ]
    
    for slider in aug_inputs[1:]:
        slider.change(fn=update_aug_preview, inputs=aug_inputs, outputs=c["aug_preview_img"])
