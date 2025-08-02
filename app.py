from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import easyocr
import uuid
import shutil
from pathlib import Path
import json
from typing import List, Dict, Any

# Import your existing functions
from plate_detection import (
    enhance_plate_image_advanced,
    enhance_plate_image_for_ocr,
    filter_plate_text_relaxed,
    process_two_line_plate,
    combine_partial_results,
    try_multiple_ocr_methods,
    process_image_enhanced_ocr
)

app = FastAPI(title="License Plate Detection API", version="1.0.0")

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory="results"), name="results")

# Global models (loaded once)
vehicle_model = None
plate_model = None
ocr_engine = None

@app.on_event("startup")
async def load_models():
    global vehicle_model, plate_model, ocr_engine
    try:
        vehicle_model = YOLO("100epoch_best.pt")
        plate_model = YOLO("license_plate_detector.pt")
        ocr_engine = easyocr.Reader(['en'])
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Model loading error: {e}")

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>License Plate Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            display: flex;
            min-height: 100vh;
            gap: 20px;
            padding: 20px;
        }
        
        .left-panel {
            flex: 1;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .right-panel {
            flex: 1;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            overflow-y: auto;
        }
        
        h1 {
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-align: center;
            font-weight: 700;
        }
        
        .upload-area {
            border: 3px dashed #4ecdc4;
            border-radius: 15px;
            padding: 60px 20px;
            text-align: center;
            margin: 30px 0;
            background: linear-gradient(45deg, rgba(78, 205, 196, 0.1), rgba(255, 107, 107, 0.1));
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover {
            border-color: #ff6b6b;
            background: linear-gradient(45deg, rgba(255, 107, 107, 0.1), rgba(78, 205, 196, 0.1));
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .upload-area p {
            font-size: 1.2em;
            color: #555;
            margin: 10px 0;
        }
        
        .upload-icon {
            font-size: 3em;
            margin-bottom: 15px;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: linear-gradient(45deg, #ffeaa7, #fab1a0);
            border-radius: 15px;
            color: #2d3436;
        }
        
        .loading p {
            font-size: 1.1em;
            font-weight: 500;
        }
        
        .error {
            color: #e74c3c;
            background: rgba(231, 76, 60, 0.1);
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            border-left: 4px solid #e74c3c;
        }
        
        .success {
            color: #27ae60;
            background: rgba(39, 174, 96, 0.1);
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            border-left: 4px solid #27ae60;
        }
        
        .results h2 {
            color: #2d3436;
            margin-bottom: 25px;
            font-size: 1.8em;
            background: linear-gradient(45deg, #a29bfe, #6c5ce7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .visualization {
            margin-bottom: 30px;
            text-align: center;
        }
        
        .visualization img {
            max-width: 100%;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        
        .visualization img:hover {
            transform: scale(1.02);
        }
        
        .vehicle-card {
            background: linear-gradient(135deg, rgba(116, 75, 162, 0.1), rgba(102, 126, 234, 0.1));
            border: none;
            margin: 20px 0;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .vehicle-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        }
        
        .vehicle-card h4 {
            color: #2d3436;
            margin-bottom: 15px;
            font-size: 1.3em;
            background: linear-gradient(45deg, #fd79a8, #fdcb6e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .vehicle-card p {
            margin: 8px 0;
            font-weight: 500;
            color: #636e72;
        }
        
        .vehicle-card strong {
            color: #2d3436;
        }
        
        .plate-display {
            background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        .plate-display h3 {
            color: white;
            font-size: 1.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .plate-number {
            font-size: 2em;
            font-weight: 900;
            letter-spacing: 3px;
            background: rgba(255,255,255,0.2);
            padding: 10px 20px;
            border-radius: 10px;
            display: inline-block;
            margin-top: 10px;
            border: 2px solid rgba(255,255,255,0.3);
        }
        
        .images {
            display: flex;
            gap: 15px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        
        .images > div {
            flex: 1;
            min-width: 200px;
        }
        
        .images img {
            width: 100%;
            max-height: 150px;
            object-fit: contain;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
            background: white;
            padding: 5px;
        }
        
        .images img:hover {
            transform: scale(1.05);
        }
        
        .images p {
            margin-bottom: 8px;
            font-weight: 600;
            color: #2d3436;
            font-size: 0.9em;
            text-align: center;
        }
        
        .download-btn {
            display: inline-block;
            background: linear-gradient(45deg, #00b894, #00cec9);
            color: white;
            padding: 15px 30px;
            text-decoration: none;
            border-radius: 25px;
            margin-top: 25px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .download-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            background: linear-gradient(45deg, #00a085, #00b7b8);
        }
        
        .spinner {
            border: 4px solid rgba(78, 205, 196, 0.3);
            border-top: 4px solid #4ecdc4;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="left-panel">
            <h1>🚗 License Plate Detection</h1>
            
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📁</div>
                <p><strong>Click here to upload an image</strong></p>
                <p>or drag and drop your file</p>
                <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="uploadFile()">
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p class="pulse">🔄 Processing image... This may take a few moments.</p>
            </div>
            
            <div id="error" class="error"></div>
            <div id="success" class="success"></div>
        </div>
        
        <div class="right-panel">
            <div id="results" class="results"></div>
        </div>
    </div>

    <script>
        async function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) return;
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('error').innerHTML = '';
            document.getElementById('success').innerHTML = '';
            document.getElementById('results').innerHTML = '';
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    displayResults(result);
                    document.getElementById('success').innerHTML = '✅ Processing completed successfully!';
                } else {
                    document.getElementById('error').innerHTML = '❌ Error: ' + result.detail;
                }
            } catch (error) {
                document.getElementById('error').innerHTML = '❌ Network error: ' + error.message;
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            
            let html = '<h2>🎯 Detection Results</h2>';
            
            if (data.visualization_url) {
                html += `<div class="visualization"><h3>📊 Visualization</h3><img src="${data.visualization_url}"></div>`;
            }
            
            if (data.results && data.results.length > 0) {
                html += '<h3>🚗 Detected Vehicles</h3>';
                
                data.results.forEach(vehicle => {
                    html += `
                        <div class="vehicle-card">
                            <h4>Vehicle ${vehicle.Vehicle_ID}: ${vehicle.Vehicle_Type}</h4>
                            <p><strong>Vehicle Confidence:</strong> ${(vehicle.Vehicle_Confidence * 100).toFixed(1)}%</p>
                            
                            <div class="plate-display">
                                <h3>🏷️ License Plate: <span class="plate-number">${vehicle.Plate_Text || 'Not detected'}</span></h3>
                            </div>
                            
                            <p><strong>Raw OCR:</strong> ${vehicle.Raw_OCR_Text || 'N/A'}</p>
                            <p><strong>OCR Confidence:</strong> ${(vehicle.OCR_Confidence * 100).toFixed(1)}%</p>
                            
                            <div class="images">
                    `;
                    
                    if (vehicle.Vehicle_Image) {
                        html += `<div><p>Vehicle Image:</p><img src="/results/${data.session_id}/${vehicle.Vehicle_Image}" alt="Vehicle"></div>`;
                    }
                    
                    if (vehicle.Plate_Image) {
                        html += `<div><p>Plate Image:</p><img src="/results/${data.session_id}/${vehicle.Plate_Image}" alt="Plate"></div>`;
                    }
                    
                    html += '</div></div>';
                });
            } else {
                html += '<p>No vehicles detected in the image.</p>';
            }
            
            if (data.csv_url) {
                html += `<div style="text-align: center;"><a href="${data.csv_url}" download class="download-btn">📥 Download CSV Results</a></div>`;
            }
            
            resultsDiv.innerHTML = html;
        }
    </script>
</body>
</html>
    """

@app.post("/detect")
async def detect_plates(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    upload_path = f"uploads/{session_id}_{file.filename}"
    output_dir = f"results/{session_id}"
    
    try:
        # Save uploaded file
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process image
        results, df = await process_image_async(upload_path, output_dir)
        
        # Prepare response
        response_data = {
            "session_id": session_id,
            "results": results,
            "csv_url": f"/results/{session_id}/detection_results.csv",
            "visualization_url": f"/results/{session_id}/detection_result.jpg"
        }
        
        # Clean up upload file
        os.remove(upload_path)
        
        return response_data
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(upload_path):
            os.remove(upload_path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

async def process_image_async(image_path: str, output_dir: str):
    """Async wrapper for the image processing function"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        raise Exception(f"Could not load image: {image_path}")
    
    # Vehicle detection
    vehicle_results = vehicle_model(frame, verbose=False, conf=0.4)[0]
    vehicle_detections = []
    
    for box in vehicle_results.boxes:
        vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        vehicle_type = vehicle_model.names[cls_id]
        vehicle_detections.append([vx1, vy1, vx2, vy2, conf, vehicle_type])
    
    # Process each vehicle
    results = []
    for i, (vx1, vy1, vx2, vy2, vehicle_conf, vehicle_type) in enumerate(vehicle_detections):
        # Crop vehicle
        vehicle_crop = frame[vy1:vy2, vx1:vx2]
        if vehicle_crop.size == 0:
            continue
        
        # Save vehicle crop
        vehicle_filename = f"vehicle_{i+1}_{vehicle_type}.jpg"
        vehicle_path = os.path.join(output_dir, vehicle_filename)
        cv2.imwrite(vehicle_path, vehicle_crop)
        
        # Plate detection
        plate_text = ""
        plate_conf = 0.0
        plate_bbox = None
        best_ocr_result = None
        
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
                        # Save original plate crop
                        plate_filename = f"plate_{i+1}_original.jpg"
                        plate_path = os.path.join(output_dir, plate_filename)
                        cv2.imwrite(plate_path, plate_crop)
                        
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
                                
                                # Save enhanced plate
                                enhanced_plate = enhance_plate_image_advanced(plate_crop)
                                enhanced_path = os.path.join(output_dir, f"plate_{i+1}_enhanced.jpg")
                                cv2.imwrite(enhanced_path, enhanced_plate)
        
        except Exception as e:
            print(f"Plate detection error: {e}")
        
        # Store results
        result = {
            'Vehicle_ID': i + 1,
            'Vehicle_Type': vehicle_type,
            'Vehicle_Confidence': round(vehicle_conf, 3),
            'Plate_Text': plate_text,
            'Plate_Confidence': round(plate_conf, 3),
            'OCR_Confidence': round(best_ocr_result[1], 3) if best_ocr_result else 0.0,
            'Raw_OCR_Text': best_ocr_result[0] if best_ocr_result else "",
            'Vehicle_BBox': (vx1, vy1, vx2, vy2),
            'Plate_BBox': plate_bbox,
            'Vehicle_Image': vehicle_filename,
            'Plate_Image': f"plate_{i+1}_original.jpg" if plate_bbox else None
        }
        results.append(result)
    
    # Create visualization
    vis_frame = frame.copy()
    
    for result in results:
        vx1, vy1, vx2, vy2 = result['Vehicle_BBox']
        vehicle_type = result['Vehicle_Type']
        plate_text = result['Plate_Text']
        raw_text = result['Raw_OCR_Text']
        
        # Draw vehicle bounding box
        cv2.rectangle(vis_frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
        
        # Draw vehicle label
        label = f"{vehicle_type}"
        cv2.putText(vis_frame, label, (vx1, vy1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw plate bounding box if available
        if result['Plate_BBox']:
            px1, py1, px2, py2 = result['Plate_BBox']
            cv2.rectangle(vis_frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
            
            # Show both raw and filtered text
            if raw_text:
                cv2.putText(vis_frame, f"Raw: {raw_text}", (px1, py1 - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            if plate_text:
                cv2.putText(vis_frame, f"Filtered: {plate_text}", (px1, py1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Save visualization
    vis_path = os.path.join(output_dir, "detection_result.jpg")
    cv2.imwrite(vis_path, vis_frame)
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "detection_results.csv")
    df.to_csv(csv_path, index=False)
    
    return results, df

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": all([vehicle_model, plate_model, ocr_engine])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)






