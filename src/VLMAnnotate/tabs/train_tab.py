import gradio as gr
from src.VLMAnnotate.ui_logic import cleanup_and_refresh_ui

def create_tab(app):
    with gr.Tab("🧠 Train Model", id="train_tab") as tab:
        gr.Markdown("## Train New Model")
        gr.Markdown("Train a new model from your CVAT Task")
        
        with gr.Column():
            cvat_projects_dropdown = gr.Dropdown(
                label="Choose CVAT Projects", 
                choices=app.get_cvat_projects(), 
                visible=False, 
                interactive=False
            )

            # Get initial CVAT tasks and select first one
            cvat_tasks_initial = app.get_cvat_tasks(cvat_projects_dropdown.value)
            cvat_initial_value = cvat_tasks_initial[0] if cvat_tasks_initial else None

            cvat_tasks_dropdown = gr.Dropdown(
                label="Choose CVAT Tasks", 
                choices=cvat_tasks_initial,
                value=cvat_initial_value,  # Select first task on load
                visible=True, 
                interactive=True
            )
            with gr.Row():
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
                
            train_config_checkbox = gr.Checkbox(
                value=False,
                label="⚙️ Set Own Training Configuration"
            )

            with gr.Group(visible=False) as train_config_group:
                gr.Markdown("### ⚙️ Training Configuration")
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
                    
                gr.Markdown("#### 📊 Dataset Split Ratios (Must sum approx to 10)")
                with gr.Row():
                    train_ratio = gr.Number(value=7, label="Train Ratio", precision=0)
                    val_ratio = gr.Number(value=2, label="Validation Ratio", precision=0)
                    test_ratio = gr.Number(value=1, label="Test Ratio", precision=0)
                
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

            # ============================================================
            # NEW: Use Own Formatted Dataset Toggle
            # ============================================================
            use_formatted_checkbox = gr.Checkbox(
                value=False,
                label="📦 Use Own Formatted Dataset",
                info="Toggle to train using your own pre-formatted dataset"
            )
            
            with gr.Group(visible=False) as formatted_dataset_group:
                formatted_dataset_dropdown_train = gr.Dropdown(
                    label="Select Formatted Dataset",
                    choices=[d["name"] for d in app.get_all_own_datasets()],
                    interactive=True
                )

            train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg", elem_id="btn")
            train_own_btn = gr.Button("🚀 Train with Uploaded Dataset", variant="primary", size="lg", elem_id="btn", visible=False)
            
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
        "cvat_tasks_dropdown": cvat_tasks_dropdown,
        "format_dropdown": format_dropdown,
        "model_dropdown": model_dropdown,
        "train_config_checkbox": train_config_checkbox,
        "train_config_group": train_config_group,
        "experiments_slider": experiments_slider,
        "imgsz_slider": imgsz_slider,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
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
        "use_formatted_checkbox": use_formatted_checkbox,
        "formatted_dataset_group": formatted_dataset_group,
        "formatted_dataset_dropdown_train": formatted_dataset_dropdown_train,
        "train_btn": train_btn,
        "train_own_btn": train_own_btn,
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
                
    def trigger_training(task_id, format_name, custom_name, model, epochs, imgsz, manual_aug, 
                                     r_train, r_val, r_test,
                                     h_h, h_s, h_v, bgr_p, 
                                     deg, trans, scl, shr, 
                                     persp, f_ud, f_lr, 
                                     mos, mix, cut, cp, 
                                     ers): 
        # 1. Format/Prepare Data
        split_ratios = (r_train, r_val, r_test)
        msg, path = app.process_cvat_task(task_id, custom_name, split_ratios, format_name=format_name)
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
            clean_id = str(task_id).split(':')[0].strip()
            project_name = f"Task_{clean_id}"
                        
        yield "🚀 Training Started... ETA should appear shortly.", stats_str, project_name
                    
        result = app.start_training(path, model, epochs, imgsz, manual_aug, format_name=format_name, **aug_args)
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

    def toggle_training_mode(checked): 
        """Toggle between CVAT task training and formatted dataset training"""
        if checked:
            return (
                gr.update(visible=False),  # cvat_tasks_dropdown
                gr.update(visible=False),  # train_btn
                gr.update(visible=True),   # formatted_dataset_group
                gr.update(visible=True)    # train_own_btn
            )
        else:
            return (
                gr.update(visible=True),   # cvat_tasks_dropdown
                gr.update(visible=True),   # train_btn
                gr.update(visible=False),  # formatted_dataset_group
                gr.update(visible=False)   # train_own_btn
            )

    def on_train_config_toggle(checked):
        """Toggle training config visibility"""
        return gr.update(visible=checked)

    def trigger_formatted_training(dataset_name, format_name, model, epochs, imgsz, manual_aug,
                                                r_train, r_val, r_test, h_h, h_s, h_v, bgr_p,
                                                deg, trans, scl, shr, persp, f_ud, f_lr,
                                                mos, mix, cut, cp, ers): 
        # 1. Get dataset path from settings
        own_datasets = app.get_all_own_datasets()
        dataset = next((d for d in own_datasets if d["name"] == dataset_name), None)
                    
        if not dataset:
            yield "❌ Dataset not found", "❌ Error", None
            return
                    
        dataset_path = dataset["path"]
                    
        # 2. Inspect dataset
        stats = app.inspect_dataset_zip(dataset_path)
        stats_str = "📊 Dataset Stats:\\n"
        if stats.get("status") != "Error":
            stats_str += f"Valid Images: {stats['images']}\\n"
            stats_str += f"Classes ({stats['classes']}): {stats['class_names']}\\n"
        else:
            stats_str += f"Error inspecting stats: {stats.get('message')}\\n"
                    
        yield f"✅ Using formatted dataset: {dataset_name}\\n\\n🚀 Training Started...", stats_str, None
                    
        # 3. Augmentation params
        aug_args = {
            'hsv_h': h_h, 'hsv_s': h_s, 'hsv_v': h_v, 'bgr': bgr_p,
            'degrees': deg, 'translate': trans, 'scale': scl, 'shear': shr,
            'perspective': persp, 'flipud': f_ud, 'fliplr': f_lr,
            'mosaic': mos, 'mixup': mix, 'cutmix': cut, 'copy_paste': cp,
            'erasing': ers
        }
                    
        project_name = dataset_name
        yield "🚀 Training Started... ETA should appear shortly.", stats_str, project_name
                    
        # 4. Start training
        result = app.start_training(dataset_path, model, epochs, imgsz, manual_aug, 
                                     format_name=format_name, **aug_args)
        yield result, stats_str, None

    def load_sample_image(task_str): 
        if not task_str: return None, None
        task_str = str(task_str) # Ensure string
        task_id = task_str.split(':')[0].strip() if ':' in task_str else task_str
        images = app.get_random_sample_images(task_id)
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

    # --- Event Handlers ---
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
            c["cvat_tasks_dropdown"], c["format_dropdown"], c["filename_input"], c["model_dropdown"], c["experiments_slider"], c["imgsz_slider"], c["manual_aug_checkbox"],
            c["train_ratio"], c["val_ratio"], c["test_ratio"],
            c["hsv_h"], c["hsv_s"], c["hsv_v"], c["bgr"],
            c["degrees"], c["translate"], c["scale"], c["shear"],
            c["perspective"], c["flipud"], c["fliplr"],
            c["mosaic"], c["mixup"], c["cutmix"], c["copy_paste"],
            c["erasing"]
        ],
        outputs=[c["training_log"], c["dataset_log"], c["current_training_project_name"]]
    )
    
    c["use_formatted_checkbox"].change(
        fn=toggle_training_mode,
        inputs=[c["use_formatted_checkbox"]],
        outputs=[c["cvat_tasks_dropdown"], c["train_btn"], c["formatted_dataset_group"], c["train_own_btn"]]
    )
    
    c["train_config_checkbox"].change(
        fn=on_train_config_toggle,
        inputs=[c["train_config_checkbox"]],
        outputs=[c["train_config_group"]]
    )
    
    c["train_own_btn"].click(
        fn=trigger_formatted_training,
        inputs=[
            c["formatted_dataset_dropdown_train"], c["format_dropdown"], c["model_dropdown"], 
            c["experiments_slider"], c["imgsz_slider"], c["manual_aug_checkbox"],
            c["train_ratio"], c["val_ratio"], c["test_ratio"],
            c["hsv_h"], c["hsv_s"], c["hsv_v"], c["bgr"],
            c["degrees"], c["translate"], c["scale"], c["shear"],
            c["perspective"], c["flipud"], c["fliplr"],
            c["mosaic"], c["mixup"], c["cutmix"], c["copy_paste"],
            c["erasing"]
        ],
        outputs=[c["training_log"], c["dataset_log"], c["current_training_project_name"]]
    )
    
    c["load_sample_btn"].click(
        fn=load_sample_image,
        inputs=[c["cvat_tasks_dropdown"]],
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
