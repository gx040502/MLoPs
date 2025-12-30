# page_datasets_v3.py - Skeleton Outline

**Total Lines**: ~1555 lines  
**Structure**: Strict 3-section organization

---

## 📁 File Header (Lines 1-9)
```python
import gradio as gr
from pathlib import Path
import os
import json
import random
import time
from typing import List, Dict, Any

def load_dataset_interface(app_interface, app):
```

---

## ⚙️ SECTION 1: ALL FUNCTION DEFINITIONS (Lines 10-618)

### CSS Styling (Lines 11-47)
- `gr.HTML(...)` - Custom CSS for buttons and sliders

### 🌐 Global Functions (Lines 49-116)
- `refresh_datasets_state()` - Get current datasets and return as state
- `refresh_all_components()` - Refresh all dataset-related components
- `cleanup_and_refresh_ui()` - Cleanup temp datasets and refresh all dropdowns

### 📤 Upload & Select Tab Functions (Lines 118-249)
- `upload_and_refresh(zip_file)` - Handle upload and refresh datasets
- `delete_dataset_handler(dataset_name)` - Handle dataset deletion
- `on_dataset_select(dataset_name)` - Handle dataset selection and video detection
- `on_annotate_click(dataset_name, video_df, interval_val)` - Handle annotate button click

### 📦 Formatted Dataset Functions (Lines 251-317)
- `upload_formatted_and_refresh(zip_file)` - Handle formatted dataset upload
- `delete_formatted_and_refresh(dataset_name)` - Handle formatted dataset deletion
- `on_formatted_dataset_select(dataset_name)` - Update path info on selection
- `navigate_to_train_with_dataset(selected_dataset_name)` - Navigate to Train tab

### 📝 VLM Annotation Tab Functions (Lines 319-394)
- `load_selected_img()` - Load selected image from dataset
- `refresh_vlm_dropdown()` - Refresh dataset dropdown
- `get_dataset_images_for_gallery()` - Get all images for gallery display
- `on_vlm_dataset_select(dataset_name)` - Handle VLM dataset selection
- `on_vlm_process_click(dataset_name, video_df, interval_val)` - Process videos
- `on_gallery_select(evt: gr.SelectData)` - Handle gallery image selection

### 🧠 Train Model Tab Functions (Lines 396-543)
- `on_format_change(format_val)` - Update model dropdown based on format
- `toggle_aug_settings(checkbox_val)` - Toggle augmentation settings visibility
- `trigger_training(...)` - Trigger CVAT task training (13 parameters)
- `check_training_status(project_name)` - Check and display training ETA
- `toggle_training_mode(checked)` - Toggle between CVAT and formatted dataset
- `on_train_config_toggle(checked)` - Toggle training config visibility
- `trigger_formatted_training(...)` - Trigger formatted dataset training (15 parameters)
- `load_sample_image(task_str)` - Load sample for augmentation preview
- `update_aug_preview(...)` - Update augmentation preview (14 parameters)
- `on_tab_select(evt: gr.SelectData)` - Cleanup preview on tab switch

### 🎱 Predict Tab Functions (Lines 545-581)
- `on_predict_tab_select()` - Refresh projects on tab select
- `on_project_change(project_name)` - Update models and test images
- `on_model_change(project_name, model_name)` - Update model details
- `on_predict(project, model, img, conf, iou)` - Run prediction
- `toggle_plots_visibility(checked)` - Toggle plots group visibility

### 🤖 Pre-Trained Models Tab Functions (Lines 583-632)
- `refresh_own_model_dropdown()` - Refresh custom models dropdown
- `handle_model_upload(model_name, upload_file)` - Handle model upload
- `handle_model_delete(model_name)` - Handle model deletion
- `display_model_details(model_name)` - Display selected model details
- `predict_own_model(model_name, image, conf, iou)` - Run prediction with custom model

### ℹ️ About Tab Functions (Lines 634-638)
- `update_dataset_dropdown_and_display(datasets)` - Update dropdown choices

---

## 🎨 SECTION 2: ALL UI COMPONENTS (Lines 640-1227)

### Global State & Header (Lines 640-631)
```python
datasets_state = gr.State(value=refresh_datasets_state())
gr.Markdown("🗂️ Data Management System")
```

### with gr.Tabs() as tabs:

#### 📤 Tab 1: Upload and Select (Lines 633-795)
**Upload Raw Dataset Section**:
- `zip_file_input` - File uploader for ZIP
- `upload_btn` - Upload & Extract button
- `upload_status` - Status markdown
- `dataset_dropdown` - Select dataset dropdown
- `status_output` - Selected dataset textbox
- `delete_btn` - Delete dataset button
- `annotate_btn` - Annotate button
- `delete_status` - Delete status markdown

**Video Frame Extraction Configuration**:
- `video_config_group` - Group (initially hidden)
- `interval_slider` - Extraction interval slider
- `video_dataframe` - Videos to process dataframe

**Upload Own Formatted Dataset Section**:
- `formatted_dataset_file` - File uploader
- `upload_formatted_btn` - Upload button
- `upload_formatted_status` - Status markdown
- `formatted_dataset_dropdown` - Select dropdown
- `formatted_dataset_path_info` - Path textbox
- `delete_formatted_btn` - Delete button
- `train_formatted_btn` - Train Dataset button

#### 📝 Tab 2: VLM Annotation (Lines 797-917)
- `vlm_dataset_dropdown` - Dataset selection dropdown
- `vlm_video_group` - Video processing group (hidden)
- `vlm_interval_slider` - Extraction interval
- `vlm_video_dataframe` - Videos dataframe
- `vlm_process_btn` - Process & Load button
- `vlm_gallery` - Dataset gallery
- `vlm_image_input` - Image input
- `vlm_text_input` - Text prompt
- `inference_format` - Format dropdown
- `vlm_confidence_slider` - Confidence threshold
- `detect_btn` - Run On Image button
- `clear_btn` - Clear button
- `prompt_output` - Parameters textbox
- `vlm_output_image` - Detection results
- `inference_btn` - Inference Full Dataset button
- `inference_output` - HTML output
- `cvat_project_dropdown` - CVAT project dropdown
- `cvat_btn` - Create CVAT Task button
- `detection_info` - Detection details
- `raw_output` - Raw results

#### 🧠 Tab 3: Train Model (Lines 919-1105)
- `cvat_projects_dropdown` - CVAT projects dropdown
- `cvat_tasks_dropdown` - CVAT tasks dropdown
- `format_dropdown` - Choose format dropdown
- `model_dropdown` - Select pre-trained model
- `train_config_checkbox` - Set own config checkbox

**Training Configuration Group**:
- `train_config_group` - Group (hidden)
- `experiments_slider` - Epochs slider
- `imgsz_slider` - Image size slider
- `train_ratio` - Train ratio number
- `val_ratio` - Validation ratio number
- `test_ratio` - Test ratio number
- `manual_aug_checkbox` - Manual augmentation checkbox

**Augmentation Settings Group**:
- `aug_settings_group` - Group (hidden)
- `aug_preview_img` - Preview image
- `load_sample_btn` - Load new sample button
- `hsv_h, hsv_s, hsv_v` - HSV sliders
- `bgr` - BGR flip probability
- `degrees, translate, scale, shear` - Geometric sliders
- `perspective, flipud, fliplr` - Transform sliders
- `mosaic, mixup, cutmix, copy_paste` - Mixing sliders
- `erasing` - Erasing slider

**Training Controls**:
- `use_formatted_checkbox` - Use own formatted dataset
- `formatted_dataset_group` - Group (hidden)
- `formatted_dataset_dropdown_train` - Select dataset
- `train_btn` - Start Training button
- `train_own_btn` - Train with Uploaded Dataset (hidden)
- `dataset_log` - Dataset detail log
- `training_log` - Training log
- `eta_output` - ETA markdown
- `current_training_project_name` - State
- `filename_input` - Custom filename

#### 🎱 Tab 4: Predict Model (Lines 1107-1195)
- `train_project_dd` - Select project dropdown
- `train_model_dd` - Select model dropdown
- `model_details` - Model details textbox
- `show_plots_checkbox` - Show analysis checkbox
- `plots_group` - Plots group (hidden)
- `model_plots_gallery` - Training analysis gallery
- `test_gallery` - Test images gallery
- `input_img` - Input image
- `conf_slider` - Confidence threshold
- `iou_slider` - IOU threshold
- `predict_btn` - Predict Image button
- `output_img` - Prediction result
- `result_details` - Detection details code

#### 🤖 Tab 5: Pre-Trained Models (Lines 1197-1255)
**Add Custom Model Accordion**:
- `model_name_input` - Model name textbox
- `model_uploader` - File uploader
- `upload_model_btn` - Upload Model button
- `upload_model_status` - Status markdown

**Model Management**:
- `own_model_dropdown` - Select model dropdown
- `own_delete_btn` - Delete Model button
- `own_model_details` - Model details code

**Prediction Interface**:
- `own_input_img` - Input image
- `own_conf_slider` - Confidence slider
- `own_iou_slider` - IOU slider
- `own_predict_btn` - Predict Image button
- `own_output_img` - Prediction result
- `own_result_details` - Detection details

#### ℹ️ Tab 6: About (Lines 1257-1277)
- Markdown content explaining features

---

## 🔗 SECTION 3: ALL EVENT HANDLERS (Lines 1229-1542)

### 📤 Upload & Select Tab Event Handlers (Lines 1232-1327)
```python
dataset_dropdown.change(...)          # Select dataset
annotate_btn.click(...)               # Navigate to VLM tab
delete_btn.click(...).then(...)       # Delete and refresh
upload_btn.click(...).then(...)       # Upload and refresh
upload_formatted_btn.click(...)       # Upload formatted
delete_formatted_btn.click(...)       # Delete formatted
formatted_dataset_dropdown.change(...) # Update path info
train_formatted_btn.click(...)        # Navigate to train tab
```

### 📝 VLM Annotation Tab Event Handlers (Lines 1329-1397)
```python
detect_btn.click(...).then(...)       # Run detection
vlm_dataset_dropdown.change(...).then(...).then(...) # Select dataset
vlm_process_btn.click(...).then(...).then(...)      # Process videos
clear_btn.click(...)                  # Clear inputs
inference_btn.click(...).then(...)    # Run inference
cvat_btn.click(...).then(...)         # Create CVAT task
vlm_text_input.submit(...)            # Submit on enter
vlm_tab.select(...).then(...).then(...) # Refresh on tab select
vlm_gallery.select(...)               # Select from gallery
```

### 🧠 Train Model Tab Event Handlers (Lines 1399-1469)
```python
format_dropdown.change(...)           # Update models
manual_aug_checkbox.change(...)       # Toggle augmentation
eta_timer.tick(...)                   # Poll training status
train_btn.click(...)                  # Start CVAT training
use_formatted_checkbox.change(...)    # Toggle mode
train_config_checkbox.change(...)     # Toggle config
train_own_btn.click(...)              # Start formatted training
load_sample_btn.click(...)            # Load preview sample
for slider in aug_inputs[1:]:         # Update preview on change
    slider.change(...)
```

### 🎱 Predict Tab Event Handlers (Lines 1471-1501)
```python
predict_tab.select(...)               # Refresh on tab select
train_project_dd.change(...)          # Update models
train_model_dd.change(...)            # Update details
predict_btn.click(...)                # Run prediction
test_gallery.select(...)              # Select test image
show_plots_checkbox.change(...)       # Toggle plots
```

### 🤖 Pre-Trained Models Tab Event Handlers (Lines 1503-1525)
```python
pre_trained_tab.select(...)           # Refresh dropdown
upload_model_btn.click(...)           # Upload model
own_model_dropdown.change(...)        # Show details
own_delete_btn.click(...)             # Delete model
own_predict_btn.click(...)            # Run prediction
```

### 🌐 Global Tab Event Handlers (Lines 1527-1537)
```python
common_inputs = []
common_outputs = [dataset_dropdown, vlm_dataset_dropdown, ...]

upload_select_tab.select(...)         # Cleanup on tab switch
train_tab.select(...)                 # Cleanup on tab switch
predict_tab.select(...)               # Cleanup on tab switch
about_tab.select(...)                 # Cleanup on tab switch
```

### 🚀 App Initialization (Lines 1539-1542)
```python
app_interface.load(
    fn=refresh_all_components,
    outputs=[datasets_state, dataset_dropdown, ...]
)
```

---

## 🏁 Main Execution Block (Lines 1544-1555)
```python
if __name__ == "__main__":
    from app import APP
    app = APP()
    
    with gr.Blocks(title="See.AI Agent") as app_interface:
        load_dataset_interface(app_interface, app)
        gr.Button("Go to VLM", link="/vlm")
    
    app_interface.launch(...)
```

---

## 📊 Summary Statistics

| Section | Line Range | Count | Description |
|---------|------------|-------|-------------|
| **Section 1: Functions** | 10-638 | ~45 functions | All helper functions organized by tab |
| **Section 2: UI Components** | 640-1227 | 80+ components | All Gradio UI elements by tab |
| **Section 3: Event Handlers** | 1229-1542 | 45+ handlers | All event bindings by tab |
| **Main Block** | 1544-1555 | 1 block | Standalone execution |

---

## 🎯 Key Benefits of 3-Section Structure

1. ✅ **Easy Navigation** - Jump to any section quickly
2. ✅ **Clear Separation** - Logic, UI, and events are distinct
3. ✅ **Maintainability** - Easy to find and modify code
4. ✅ **No Mixing** - Pure functional organization
5. ✅ **Scalability** - Easy to add new features

---

## 🔍 Quick Reference

**To find a function**: Check Section 1 (lines 10-638)  
**To find a UI component**: Check Section 2 (lines 640-1227)  
**To find an event handler**: Check Section 3 (lines 1229-1542)  
**To modify training logic**: See Train Tab Functions (lines 396-543)  
**To modify VLM inference**: See VLM Tab Functions (lines 319-394)
