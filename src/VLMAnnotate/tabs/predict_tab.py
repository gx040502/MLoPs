import gradio as gr

def create_tab(app):
    with gr.Tab("🎱 Predict Model") as tab:
        # UI components only
        gr.Markdown("## Predict Model")
        gr.Markdown("Select a trained model and run inference on custom images.")
        
        with gr.Column():
            # 1. Model Selection
            # Get initial projects and set default value
            initial_projects = app.list_trained_projects()
            initial_project_value = initial_projects[0] if initial_projects else None
            
            # Get initial models if a project is selected
            initial_models = []
            initial_model_value = None
            if initial_project_value:
                initial_models = app.list_trained_models(initial_project_value)
                initial_model_value = initial_models[0] if initial_models else None
            
            with gr.Row():
                train_project_dd = gr.Dropdown(
                    label="1. Select Project",
                    choices=initial_projects,
                    value=initial_project_value,  # Set initial value
                    interactive=True,
                    elem_id="train_project_dd"
                )
                
                train_model_dd = gr.Dropdown(
                    label="2. Select Model",
                    choices=initial_models,  # Populate with initial models
                    value=initial_model_value,  # Set initial model value
                    interactive=True,
                    elem_id="train_model_dd"
                )
            
            model_details = gr.Textbox(
                label="Model Details",
                lines=3,
                interactive=False
            )
            
            show_plots_checkbox = gr.Checkbox(
                value=False,
                label="Show Model Training Analysis"
            )

            with gr.Group(visible=False) as plots_group:
                model_plots_gallery = gr.Gallery(
                    label="Model Training Analysis", 
                    show_label=True, 
                    elem_id="model_plots",
                    columns=[4],
                    rows=[1],
                    height=200,
                    allow_preview=True,
                    object_fit="contain",
                    interactive=False
                )

            # 2. Prediction Interface
            gr.Markdown("### 📂 Select Test Image")
            test_gallery = gr.Gallery(
                label="Test Images", 
                show_label=False, 
                elem_id="test_gallery",
                columns=[4],
                rows=[1],
                height=150,
                allow_preview=True,
                interactive=True
            )
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🖼️ Run Prediction")
                    input_img = gr.Image(
                        label="Input Image", 
                        type="pil",
                        height=400
                    )
                        
                    conf_slider = gr.Slider(
                        minimum=0.01, maximum=1.0, value=0.25, 
                        step=0.01, label="Confidence Threshold"
                    )
                    
                    iou_slider = gr.Slider(
                        minimum=0.01, maximum=1.0, value=0.45, 
                        step=0.01, label="IOU Threshold"
                    )
                        
                    predict_btn = gr.Button("🚀 Predict Image", variant="primary", elem_id="btn")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Prediction Result")
                    output_img = gr.Image(label="Prediction Result", type="pil", height=400)
                    result_details = gr.Code(label="Detection Details", language="json", elem_id="detection_details_code", lines=10)
    
    return {
        "tab": tab,
        "train_project_dd": train_project_dd,
        "train_model_dd": train_model_dd,
        "model_details": model_details,
        "show_plots_checkbox": show_plots_checkbox,
        "plots_group": plots_group,
        "model_plots_gallery": model_plots_gallery,
        "test_gallery": test_gallery,
        "input_img": input_img,
        "conf_slider": conf_slider,
        "iou_slider": iou_slider,
        "predict_btn": predict_btn,
        "output_img": output_img,
        "result_details": result_details
    }

def setup_events(app, components, all_components):
    c = components
    
    # Internal logic
    def on_predict_tab_select():
        """Refreshes the project dropdown when tab is selected."""
        app.cleanup_preview()
        projects = app.list_trained_projects()
        return gr.update(choices=projects), gr.update(choices=[], value=None)

    def on_project_change(project_name):
        # Update model dropdown
        models = app.list_trained_models(project_name)
        # Update test images
        test_imgs = app.get_test_images(project_name)
        return gr.update(choices=models, value=None), "", gr.update(value=test_imgs)

    def on_model_change(project_name, model_name):
        # Update details
        details = app.get_model_details(project_name, model_name)
        # Update plots
        plots = app.get_model_plots(project_name, model_name)
        
        # Detect if it's a classification model
        is_classification = "-cls" in model_name.lower() if model_name else False
        
        # Hide conf/IOU sliders for classification models
        return (
            details, 
            gr.update(value=plots),
            gr.update(visible=not is_classification),  # conf_slider
            gr.update(visible=not is_classification)   # iou_slider
        )

    def on_predict(project, model, img, conf, iou):
        return app.predict_with_model(project, model, img, conf, iou)

    def toggle_plots_visibility(checked):
        return gr.update(visible=checked)
    
    def on_gallery_select(evt: gr.SelectData):
        """Handle gallery selection to load image into input"""
        return evt.value["image"]["path"]

    # --- Event Handlers ---
    c["tab"].select(
        fn=on_predict_tab_select, 
        outputs=[c["train_project_dd"], c["train_model_dd"]]
    )
    
    c["train_project_dd"].change(
        fn=on_project_change, 
        inputs=[c["train_project_dd"]], 
        outputs=[c["train_model_dd"], c["model_details"], c["test_gallery"]]
    )
    
    c["train_model_dd"].change(
        fn=on_model_change,
        inputs=[c["train_project_dd"], c["train_model_dd"]],
        outputs=[c["model_details"], c["model_plots_gallery"], c["conf_slider"], c["iou_slider"]]
    )
    
    c["predict_btn"].click(
        fn=on_predict,
        inputs=[c["train_project_dd"], c["train_model_dd"], c["input_img"], c["conf_slider"], c["iou_slider"]],
        outputs=[c["output_img"], c["result_details"]]
    )
    
    c["test_gallery"].select(
        fn=on_gallery_select,
        outputs=[c["input_img"]]
    )
    
    c["show_plots_checkbox"].change(
        fn=toggle_plots_visibility, 
        inputs=c["show_plots_checkbox"],
        outputs=c["plots_group"]
    )
