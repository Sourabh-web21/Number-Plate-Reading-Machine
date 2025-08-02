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
    <html lang="en">
    <head>
        <title>AI License Plate Detection System</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            :root {
                --primary: #667eea;
                --secondary: #764ba2;
                --accent: #f093fb;
                --success: #4caf50;
                --error: #f44336;
                --warning: #ff9800;
                --dark: #2c3e50;
                --light: #ecf0f1;
                --shadow: 0 10px 30px rgba(0,0,0,0.1);
                --shadow-hover: 0 20px 60px rgba(0,0,0,0.2);
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 50%, var(--accent) 100%);
                min-height: 100vh;
                overflow-x: hidden;
            }

            .background-animation {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: -1;
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 50%, var(--accent) 100%);
                background-size: 400% 400%;
                animation: gradientShift 15s ease infinite;
            }

            @keyframes gradientShift {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            .container {
                max-width: 100%;
                height: 100vh;
                display: flex;
                flex-direction: column;
                position: relative;
                z-index: 1;
            }

            .app-header {
                text-align: center;
                padding: 5px;
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(20px);
                border-bottom: 1px solid rgba(255,255,255,0.2);
                height: 60px;
            }

            .app-title {
                font-size: 1.1rem;
                font-weight: 700;
                color: white;
                margin-bottom: 3px;
                text-shadow: 0 4px 8px rgba(0,0,0,0.3);
            }

            .app-subtitle {
                font-size: 0.9rem;
                color: rgba(255,255,255,0.9);
                font-weight: 300;
            }

            .split-container {
                display: flex;
                flex: 1;
                gap: 0;
                padding: 0;
                height: calc(100vh - 60px);
            }

            .left-panel, .right-panel {
                flex: 1;
                width: 50%;
                background: rgba(255,255,255,0.95);
                backdrop-filter: blur(20px);
                border-radius: 0;
                padding: 30px;
                box-shadow: none;
                border: none;
                overflow-y: auto;
            }

            .left-panel {
                border-right: 2px solid rgba(255,255,255,0.3);
            }

            .panel-title {
                font-size: 1.5rem;
                font-weight: 600;
                color: var(--dark);
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .upload-area {
                border: 3px dashed #ddd;
                border-radius: 15px;
                padding: 40px 20px;
                text-align: center;
                cursor: pointer;
                background: linear-gradient(135deg, #f8f9ff 0%, #fff 100%);
                transition: all 0.3s ease;
                margin-bottom: 20px;
            }

            .upload-area:hover {
                border-color: var(--primary);
                background: linear-gradient(135deg, #f0f4ff 0%, #fff 100%);
                transform: scale(1.02);
            }

            .upload-area.dragover {
                border-color: var(--success);
                background: linear-gradient(135deg, #f0fff4 0%, #fff 100%);
                transform: scale(1.05);
            }

            .upload-icon {
                font-size: 3rem;
                color: var(--primary);
                margin-bottom: 15px;
                transition: all 0.3s ease;
            }

            .upload-area:hover .upload-icon {
                transform: scale(1.1) rotate(5deg);
                color: var(--secondary);
            }

            .upload-text {
                font-size: 1.2rem;
                font-weight: 600;
                color: var(--dark);
                margin-bottom: 8px;
            }

            .upload-subtext {
                color: #666;
                font-size: 0.9rem;
            }

            #fileInput {
                display: none;
            }

            .progress-container {
                display: none;
                margin-top: 20px;
                background: rgba(255,255,255,0.95);
                backdrop-filter: blur(20px);
                border-radius: 15px;
                padding: 20px;
                box-shadow: var(--shadow);
            }

            .progress-header {
                display: flex;
                align-items: center;
                gap: 15px;
                margin-bottom: 15px;
            }

            .progress-spinner {
                width: 30px;
                height: 30px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid var(--primary);
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .progress-text {
                font-weight: 600;
                color: var(--dark);
            }

            .progress-bar {
                width: 100%;
                height: 8px;
                background: #f0f0f0;
                border-radius: 4px;
                overflow: hidden;
            }

            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, var(--primary), var(--secondary));
                border-radius: 4px;
                width: 0%;
                animation: progressAnimation 3s ease-in-out infinite;
            }

            @keyframes progressAnimation {
                0%, 100% { width: 0%; }
                50% { width: 100%; }
            }

            .alert {
                padding: 15px;
                border-radius: 10px;
                margin: 15px 0;
                font-weight: 500;
                display: none;
                animation: slideIn 0.5s ease-out;
            }

            .alert-success {
                background: linear-gradient(135deg, #d4edda, #c3e6cb);
                color: #155724;
                border-left: 5px solid var(--success);
            }

            .alert-error {
                background: linear-gradient(135deg, #f8d7da, #f5c6cb);
                color: #721c24;
                border-left: 5px solid var(--error);
            }

            .results-container {
                display: none;
            }

            .visualization-card {
                margin-bottom: 20px;
            }

            .visualization-card img {
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                cursor: pointer;
                transition: all 0.3s ease;
            }

            .visualization-card img:hover {
                transform: scale(1.02);
            }

            .vehicles-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 20px;
            }

            .vehicle-card {
                background: rgba(255,255,255,0.95);
                backdrop-filter: blur(20px);
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                border: 1px solid rgba(255,255,255,0.2);
                transition: all 0.3s ease;
            }

            .vehicle-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }

            .vehicle-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }

            .vehicle-title {
                font-weight: 600;
                color: var(--dark);
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .vehicle-type {
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                color: white;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 500;
            }

            .plate-display {
                background: linear-gradient(135deg, #2c3e50, #34495e);
                color: white;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                font-family: 'Courier New', monospace;
                font-size: 1.2rem;
                font-weight: bold;
                letter-spacing: 2px;
                margin-bottom: 15px;
                border: 3px solid #fff;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }

            .plate-display.no-plate {
                background: linear-gradient(135deg, #95a5a6, #7f8c8d);
                font-style: italic;
                letter-spacing: normal;
            }

            .stats-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin-bottom: 15px;
            }

            .stat-item {
                text-align: center;
                padding: 8px;
                background: rgba(102, 126, 234, 0.1);
                border-radius: 8px;
            }

            .stat-label {
                font-size: 0.8rem;
                color: #666;
                margin-bottom: 4px;
            }

            .stat-value {
                font-weight: 600;
                color: var(--dark);
            }

            .images-section {
                margin-top: 15px;
            }

            .images-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
            }

            .image-container {
                text-align: center;
            }

            .image-label {
                font-size: 0.8rem;
                color: #666;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 5px;
            }

            .image-wrapper img {
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                cursor: pointer;
                transition: all 0.3s ease;
            }

            .image-wrapper img:hover {
                transform: scale(1.05);
            }

            .no-vehicles {
                text-align: center;
                padding: 40px;
                color: #666;
                font-size: 1.1rem;
            }

            .no-vehicles i {
                font-size: 3rem;
                margin-bottom: 15px;
                color: #ccc;
            }

            /* Responsive Design */
            @media (max-width: 1024px) {
                .split-container {
                    flex-direction: row;
                    height: calc(100vh - 60px);
                }
                
                .left-panel, .right-panel {
                    width: 50%;
                    height: 100%;
                    min-height: auto;
                }
                
                .stats-grid {
                    grid-template-columns: 1fr;
                }
                
                .images-grid {
                    grid-template-columns: 1fr;
                }
            }

            @media (max-width: 768px) {
                .container {
                    height: 100vh;
                }
                
                .split-container {
                    padding: 0;
                    gap: 0;
                }
                
                .left-panel, .right-panel {
                    padding: 15px;
                }
                
                .app-title {
                    font-size: 1.3rem;
                }
                
                .app-header {
                    padding: 10px;
                    height: 80px;
                }
                
                .split-container {
                    height: calc(100vh - 80px);
                }
            }

            /* Loading Animation */
            .loading-dots {
                display: inline-block;
            }

            .loading-dots::after {
                content: '';
                animation: dots 2s infinite;
            }

            @keyframes dots {
                0%, 20% { content: ''; }
                40% { content: '.'; }
                60% { content: '..'; }
                80%, 100% { content: '...'; }
            }
        </style>
    </head>
    <body>
        <div class="background-animation"></div>

        <div class="container">
            <div class="app-header">
                <h1 class="app-title">🚗 AI License Plate Detection</h1>
                <p class="app-subtitle">Advanced Computer Vision • Real-time Processing • High Accuracy</p>
            </div>
            
            <div class="split-container">
                <!-- Left Panel: Upload & Controls -->
                <div class="left-panel">
                    <div class="panel-title">
                        <i class="fas fa-cloud-upload-alt"></i>
                        Upload & Process
                    </div>
                    
                    <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                        <i class="fas fa-cloud-upload-alt upload-icon"></i>
                        <div class="upload-text">Drop image here or click to browse</div>
                        <div class="upload-subtext">Supports JPG, PNG, WEBP • Max 10MB</div>
                        <input type="file" id="fileInput" accept="image/*" onchange="uploadFile()">
                    </div>
                    
                    <div class="progress-container" id="progressContainer">
                        <div class="progress-header">
                            <div class="progress-spinner"></div>
                            <div class="progress-text">Processing image<span class="loading-dots"></span></div>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill"></div>
                        </div>
                        <p style="text-align: center; color: #666; margin-top: 10px; font-size: 0.9rem;">
                            AI is analyzing vehicles and reading license plates
                        </p>
                    </div>
                    
                    <div id="alertSuccess" class="alert alert-success">
                        <i class="fas fa-check-circle" style="margin-right: 10px;"></i>
                        <span id="successMessage"></span>
                    </div>
                    
                    <div id="alertError" class="alert alert-error">
                        <i class="fas fa-exclamation-triangle" style="margin-right: 10px;"></i>
                        <span id="errorMessage"></span>
                    </div>
                </div>

                <!-- Right Panel: Results -->
                <div class="right-panel">
                    <div class="panel-title">
                        <i class="fas fa-chart-line"></i>
                        Detection Results
                    </div>
                    
                    <div id="resultsContainer" class="results-container">
                        <div style="text-align: center; padding: 60px 20px; color: #999;">
                            <i class="fas fa-image" style="font-size: 4rem; margin-bottom: 20px; opacity: 0.3;"></i>
                            <div style="font-size: 1.2rem; margin-bottom: 10px;">No image processed yet</div>
                            <div style="font-size: 1rem;">Upload an image to see detection results</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let isProcessing = false;

            async function uploadFile() {
                if (isProcessing) return;
                
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                
                if (!file) return;
                
                // Validate file
                if (!file.type.startsWith('image/')) {
                    showAlert('error', 'Please select a valid image file');
                    return;
                }
                
                if (file.size > 10 * 1024 * 1024) {
                    showAlert('error', 'File size must be less than 10MB');
                    return;
                }
                
                isProcessing = true;
                showProgress();
                hideAlerts();
                clearResults();
                
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
                        showAlert('success', `Successfully processed! Found ${result.results.length} vehicle(s)`);
                    } else {
                        showAlert('error', result.detail || 'Processing failed');
                    }
                } catch (error) {
                    showAlert('error', 'Network error: ' + error.message);
                } finally {
                    hideProgress();
                    isProcessing = false;
                }
            }
            
            function showProgress() {
                document.getElementById('progressContainer').style.display = 'block';
            }
            
            function hideProgress() {
                document.getElementById('progressContainer').style.display = 'none';
            }
            
            function showAlert(type, message) {
                hideAlerts();
                const alertId = type === 'success' ? 'alertSuccess' : 'alertError';
                const messageId = type === 'success' ? 'successMessage' : 'errorMessage';
                
                document.getElementById(messageId).textContent = message;
                document.getElementById(alertId).style.display = 'block';
                
                setTimeout(() => {
                    document.getElementById(alertId).style.display = 'none';
                }, 5000);
            }
            
            function hideAlerts() {
                document.getElementById('alertSuccess').style.display = 'none';
                document.getElementById('alertError').style.display = 'none';
            }
            
            function clearResults() {
                const container = document.getElementById('resultsContainer');
                container.innerHTML = `
                    <div style="text-align: center; padding: 60px 20px; color: #999;">
                        <i class="fas fa-cog fa-spin" style="font-size: 4rem; margin-bottom: 20px; opacity: 0.3;"></i>
                        <div style="font-size: 1.2rem; margin-bottom: 10px;">Processing...</div>
                        <div style="font-size: 1rem;">Please wait while we analyze your image</div>
                    </div>
                `;
                container.style.display = 'block';
            }
            
            function displayResults(data) {
                const container = document.getElementById('resultsContainer');
                
                let html = '';
                
                if (data.visualization_url) {
                    html += `
                        <div class="visualization-card">
                            <h3 style="margin-bottom: 15px; color: var(--dark); display: flex; align-items: center; gap: 10px; font-size: 1.1rem;">
                                <i class="fas fa-eye"></i>
                                Detection Visualization
                                ${data.csv_url ? `<a href="${data.csv_url}" download style="margin-left: auto; background: var(--primary); color: white; padding: 5px 10px; border-radius: 5px; text-decoration: none; font-size: 0.8rem;">
                                    <i class="fas fa-download"></i> CSV
                                </a>` : ''}
                            </h3>
                            <img src="${data.visualization_url}" alt="Detection Results" onclick="openImageModal(this.src)">
                        </div>
                    `;
                }
                
                if (data.results && data.results.length > 0) {
                    html += '<div class="vehicles-grid">';
                    
                    data.results.forEach((vehicle, index) => {
                        const plateText = vehicle.Plate_Text || 'Not Detected';
                        const plateClass = vehicle.Plate_Text ? '' : 'no-plate';
                        
                        html += `
                            <div class="vehicle-card" style="animation-delay: ${index * 0.1}s">
                                <div class="vehicle-header">
                                    <div class="vehicle-title">
                                        <i class="fas fa-car"></i>
                                        Vehicle ${vehicle.Vehicle_ID}
                                    </div>
                                    <div class="vehicle-type">${vehicle.Vehicle_Type}</div>
                                </div>
                                
                                <div class="plate-display ${plateClass}">
                                    ${plateText}
                                </div>
                                
                                <div class="stats-grid">
                                    <div class="stat-item">
                                        <div class="stat-label">Vehicle Conf.</div>
                                        <div class="stat-value">${(vehicle.Vehicle_Confidence * 100).toFixed(1)}%</div>
                                    </div>
                                    <div class="stat-item">
                                        <div class="stat-label">Plate Conf.</div>
                                        <div class="stat-value">${(vehicle.Plate_Confidence * 100).toFixed(1)}%</div>
                                    </div>
                                    <div class="stat-item">
                                        <div class="stat-label">OCR Conf.</div>
                                        <div class="stat-value">${(vehicle.OCR_Confidence * 100).toFixed(1)}%</div>
                                    </div>
                                    <div class="stat-item">
                                        <div class="stat-label">Raw OCR</div>
                                        <div class="stat-value" style="font-size: 0.8rem;">${vehicle.Raw_OCR_Text || 'N/A'}</div>
                                    </div>
                                </div>
                                
                                <div class="images-section">
                                    <div class="images-grid">
                        `;
                        
                        if (vehicle.Vehicle_Image) {
                            const vehicleImagePath = `/results/${data.session_id}/${vehicle.Vehicle_Image}`;
                            html += `
                                <div class="image-container">
                                    <div class="image-label">
                                        <i class="fas fa-car"></i>
                                        Vehicle
                                    </div>
                                    <div class="image-wrapper">
                                        <img src="${vehicleImagePath}" alt="Vehicle" onclick="openImageModal(this.src)">
                                    </div>
                                </div>
                            `;
                        }
                        
                        if (vehicle.Plate_Image) {
                            const plateImagePath = `/results/${data.session_id}/${vehicle.Plate_Image}`;
                            html += `
                                <div class="image-container">
                                    <div class="image-label">
                                        <i class="fas fa-id-card"></i>
                                        Plate
                                    </div>
                                    <div class="image-wrapper">
                                        <img src="${plateImagePath}" alt="License Plate" onclick="openImageModal(this.src)">
                                    </div>
                                </div>
                            `;
                        }
                        
                        html += '</div></div></div>';
                    });
                    
                    html += '</div>';
                } else {
                    html += `
                        <div class="no-vehicles">
                            <i class="fas fa-search"></i>
                            <div>No vehicles detected</div>
                            <p style="margin-top: 10px; font-size: 0.9rem; opacity: 0.7;">
                                Try uploading a clearer image with visible vehicles
                            </p>
                        </div>
                    `;
                }
                
                container.innerHTML = html;
                container.style.display = 'block';
            }
            
            function openImageModal(src) {
                window.open(src, '_blank');
            }
            
            // Enhanced drag and drop
            const uploadArea = document.querySelector('.upload-area');
            
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, preventDefaults, false);
                document.body.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            ['dragenter', 'dragover'].forEach(eventName => {
                uploadArea.addEventListener(eventName, highlight, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, unhighlight, false);
            });
            
            function highlight(e) {
                uploadArea.classList.add('dragover');
            }
            
            function unhighlight(e) {
                uploadArea.classList.remove('dragover');
            }
            
            uploadArea.addEventListener('drop', handleDrop, false);
            
            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                
                if (files.length > 0) {
                    document.getElementById('fileInput').files = files;
                    uploadFile();
                }
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
            plate_result = plate_model(vehicle_crop, verbose=False, conf=0.3)[0]
            
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
                    plate_bbox = (vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2)
                    
                    # Crop plate
                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                    if plate_crop.size > 0:
                        # Save original plate crop
                        plate_filename = f"plate_{i+1}_original.jpg"
                        plate_path = os.path.join(output_dir, plate_filename)
                        cv2.imwrite(plate_path, plate_crop)
                        
                        # Try multiple OCR methods
                        best_ocr_result = try_multiple_ocr_methods(plate_crop, ocr_engine)
                        
                        if best_ocr_result:
                            raw_text = best_ocr_result[0]
                            filtered_text = filter_plate_text_relaxed(raw_text)
                            
                            # If filtering fails but we have raw text, use raw text
                            if filtered_text:
                                plate_text = filtered_text
                            elif raw_text and len(raw_text.strip()) >= 2:
                                plate_text = raw_text.strip().upper()
                                print(f"   Using raw OCR text: '{plate_text}'")
                            
                            if plate_text:  # If we have any plate text
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










