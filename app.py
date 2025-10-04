import gradio as gr
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import easyocr
import os
from pathlib import Path
from PIL import Image
import tempfile
import shutil

# Import your existing functions
from plate_detection import (
    enhance_plate_image_advanced,
    enhance_plate_image_for_ocr,
    filter_plate_text_relaxed,
    try_multiple_ocr_methods
)

# Create directories
os.makedirs("results", exist_ok=True)

# Global models
vehicle_model = None
plate_model = None
ocr_engine = None

def load_models():
    """Load all required models"""
    global vehicle_model, plate_model, ocr_engine
    try:
        vehicle_model = YOLO("100epoch_best.pt")
        plate_model = YOLO("license_plate_detector.pt")
        ocr_engine = easyocr.Reader(['en'])
        print("✅ Models loaded successfully")
        return "✅ Models loaded successfully"
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return f"❌ Model loading error: {e}"

def process_image(image):
    """Process uploaded image and detect license plates"""
    if image is None:
        return None, "Please upload an image", None
    
    try:
        # Convert PIL Image to numpy array (BGR format for OpenCV)
        if isinstance(image, Image.Image):
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame = image
        
        # Vehicle detection
        vehicle_results = vehicle_model(frame, verbose=False, conf=0.4)[0]
        vehicle_detections = []
        
        for box in vehicle_results.boxes:
            vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            vehicle_type = vehicle_model.names[cls_id]
            vehicle_detections.append([vx1, vy1, vx2, vy2, conf, vehicle_type])
        
        if not vehicle_detections:
            vis_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return vis_frame_rgb, "No vehicles detected in the image", None
        
        # Process each vehicle
        results = []
        vis_frame = frame.copy()
        
        for i, (vx1, vy1, vx2, vy2, vehicle_conf, vehicle_type) in enumerate(vehicle_detections):
            # Crop vehicle
            vehicle_crop = frame[vy1:vy2, vx1:vx2]
            if vehicle_crop.size == 0:
                continue
            
            # Initialize plate detection variables
            plate_text = ""
            plate_conf = 0.0
            plate_bbox = None
            best_ocr_result = None
            raw_text = ""
            
            try:
                plate_result = plate_model(vehicle_crop, verbose=False, conf=0.2)[0]
                
                if plate_result.boxes:
                    # Get the best plate detection
                    best_plate = None
                    best_conf = 0
                    
                    for box in plate_result.boxes:
                        px1, py1, px2, py2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        if conf > best_conf:
                            best_conf = conf
                            best_plate = [px1, py1, px2, py2]

                    if best_plate:
                        px1, py1, px2, py2 = best_plate
                        plate_conf = best_conf
                        
                        # Expand crop area by 10% on each side
                        crop_expand = 0.1
                        w_expand = int((px2 - px1) * crop_expand)
                        h_expand = int((py2 - py1) * crop_expand)
                        
                        px1 = max(0, px1 - w_expand)
                        py1 = max(0, py1 - h_expand)
                        px2 = min(vehicle_crop.shape[1], px2 + w_expand)
                        py2 = min(vehicle_crop.shape[0], py2 + h_expand)
                        
                        plate_bbox = (vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2)
                        
                        # Crop plate with expanded area
                        plate_crop = vehicle_crop[py1:py2, px1:px2]
                        if plate_crop.size > 0:
                            # Try multiple OCR methods with enhanced preprocessing
                            best_ocr_result = try_multiple_ocr_methods(plate_crop, ocr_engine)
                            
                            # If first attempt fails or gives partial results, try aggressive methods
                            if not best_ocr_result or len(best_ocr_result[0]) < 4:
                                # Try with enhanced preprocessing
                                enhanced_plate = enhance_plate_image_for_ocr(plate_crop)
                                enhanced_result = try_multiple_ocr_methods(enhanced_plate, ocr_engine)
                                
                                if enhanced_result and (not best_ocr_result or len(enhanced_result[0]) > len(best_ocr_result[0])):
                                    best_ocr_result = enhanced_result
                            
                            # Try different scales if still not good
                            if not best_ocr_result or len(best_ocr_result[0]) < 4:
                                for scale in [2.0, 3.0, 1.5]:
                                    h, w = plate_crop.shape[:2]
                                    scaled = cv2.resize(plate_crop, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
                                    scaled_result = try_multiple_ocr_methods(scaled, ocr_engine)
                                    
                                    if scaled_result and (not best_ocr_result or len(scaled_result[0]) > len(best_ocr_result[0])):
                                        best_ocr_result = scaled_result
                                        break
                            
                            if best_ocr_result:
                                raw_text = best_ocr_result[0]
                                filtered_text = filter_plate_text_relaxed(raw_text)
                                
                                if filtered_text:
                                    plate_text = filtered_text
            
            except Exception as e:
                print(f"Plate detection error: {e}")
            
            # Draw vehicle bounding box (green)
            cv2.rectangle(vis_frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 3)
            
            # Draw vehicle label
            label = f"{vehicle_type} ({vehicle_conf:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(vis_frame, (vx1, vy1 - label_size[1] - 10), 
                         (vx1 + label_size[0], vy1), (0, 255, 0), -1)
            cv2.putText(vis_frame, label, (vx1, vy1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Draw plate bounding box if available (blue)
            if plate_bbox:
                px1, py1, px2, py2 = plate_bbox
                cv2.rectangle(vis_frame, (px1, py1), (px2, py2), (255, 0, 0), 3)
                
                # Create background for text
                text_y = py1 - 40
                if text_y < 0:
                    text_y = py2 + 20
                
                # Show both raw and filtered text
                if raw_text:
                    raw_label = f"Raw: {raw_text}"
                    raw_size = cv2.getTextSize(raw_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(vis_frame, (px1, text_y - raw_size[1] - 5), 
                                (px1 + raw_size[0], text_y), (255, 255, 0), -1)
                    cv2.putText(vis_frame, raw_label, (px1, text_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    text_y += 25
                
                if plate_text:
                    plate_label = f"Plate: {plate_text}"
                    plate_size = cv2.getTextSize(plate_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(vis_frame, (px1, text_y - plate_size[1] - 5), 
                                (px1 + plate_size[0], text_y), (255, 0, 0), -1)
                    cv2.putText(vis_frame, plate_label, (px1, text_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Store results
            results.append({
                'Vehicle_ID': i + 1,
                'Vehicle_Type': vehicle_type,
                'Vehicle_Confidence': round(vehicle_conf, 3),
                'Plate_Text': plate_text,
                'Plate_Confidence': round(plate_conf, 3),
                'OCR_Confidence': round(best_ocr_result[1], 3) if best_ocr_result else 0.0,
                'Raw_OCR_Text': raw_text
            })
        
        # Convert BGR to RGB for display
        vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
        
        # Create summary text
        summary = f"### 🎯 Detection Results\n\n"
        summary += f"**Total Vehicles Detected:** {len(results)}\n\n"
        
        for result in results:
            summary += f"---\n"
            summary += f"**Vehicle {result['Vehicle_ID']}:** {result['Vehicle_Type']}\n"
            summary += f"- **Vehicle Confidence:** {result['Vehicle_Confidence']*100:.1f}%\n"
            summary += f"- **License Plate:** `{result['Plate_Text'] or 'Not detected'}`\n"
            summary += f"- **Plate Confidence:** {result['Plate_Confidence']*100:.1f}%\n"
            summary += f"- **OCR Confidence:** {result['OCR_Confidence']*100:.1f}%\n"
            summary += f"- **Raw OCR Text:** {result['Raw_OCR_Text'] or 'N/A'}\n\n"
        
        # Create DataFrame for download
        df = pd.DataFrame(results)
        csv_path = "results/detection_results.csv"
        df.to_csv(csv_path, index=False)
        
        return vis_frame_rgb, summary, csv_path
        
    except Exception as e:
        import traceback
        error_msg = f"Error processing image: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg, None

# Load models on startup
print("Loading models...")
status = load_models()
print(status)

# Create Gradio interface with custom theme
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
#component-0 {
    max-width: 1400px;
    margin: auto;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="License Plate Detection") as demo:
    gr.Markdown("""
    # 🚗 License Plate Detection System
    
    Upload an image of vehicles to detect license plates using advanced computer vision and OCR.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📤 Upload Vehicle Image",
                type="pil",
                height=400
            )
            
            process_btn = gr.Button(
                "🔍 Detect License Plates",
                variant="primary",
                size="lg"
            )
            
        
        
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="📊 Detection Results",
                type="numpy",
                height=400
            )
            
            output_text = gr.Markdown(label="Detection Summary")
            
            output_csv = gr.File(
                label="📥 Download Results (CSV)"
            )
    
    # Examples section (optional - add your own example images)
    gr.Markdown("""
    ---
    ### 🔧 Technical Details:
    - **Vehicle Detection Model:** YOLOv8 (100 epochs custom trained)
    - **Plate Detection Model:** YOLOv8 (license plate specialized)
    - **OCR Engine:** EasyOCR with advanced preprocessing
    - **Preprocessing:** CLAHE, sharpening, multi-scale enhancement
    - **Text Filtering:** Indian license plate format validation
    """)
    
    # Connect the button to the processing function
    process_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[output_image, output_text, output_csv]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )