import os
import cv2
import numpy as np
import pandas as pd
import difflib
import subprocess
import urllib.request
import torch
from ultralytics import YOLO
import easyocr
import re

# === Enhanced OCR Functions ===
def enhance_plate_image_advanced(img):
    """Enhanced plate image preprocessing for difficult cases"""
    if img.size == 0:
        return img

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Upscale significantly for small plates
    height, width = gray.shape
    if height < 60 or width < 120:
        scale_factor = max(3, 300 // min(height, width))
        gray = cv2.resize(gray, (width * scale_factor, height * scale_factor), 
                         interpolation=cv2.INTER_CUBIC)

    # Multiple enhancement techniques
    enhanced_versions = []
    
    # Version 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_versions.append(clahe.apply(gray))
    
    # Version 2: Gamma correction
    gamma = 1.5
    gamma_corrected = np.array(255 * (gray / 255) ** gamma, dtype='uint8')
    enhanced_versions.append(gamma_corrected)
    
    # Version 3: Bilateral filter + sharpening
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(bilateral, -1, kernel)
    enhanced_versions.append(sharpened)
    
    # Version 4: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    enhanced_versions.append(morph)
    
    # Version 5: Threshold variations
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_versions.append(thresh1)
    
    _, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    enhanced_versions.append(thresh2)
    
    # Return the original enhanced version (can be modified to return best)
    return enhanced_versions[0]

def enhance_plate_image_for_ocr(img):
    """Preprocess plate image for best OCR results, including upscaling."""
    if img.size == 0:
        return img

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Upscale if small
    height, width = gray.shape
    if height < 50 or width < 100:
        scale_factor = max(2, 300 // min(height, width))
        gray = cv2.resize(gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)

    # Mild CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light sharpening (reduced strength)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened

def filter_plate_text_relaxed(text):
    """Ultra-relaxed filtering that accepts partial plates like TM-01"""
    if not text:
        return ""
    
    # Clean text
    text = text.strip().upper()
    
    # Remove problematic OCR artifacts but keep hyphens and spaces
    text = re.sub(r'[^\w\s-]', '', text)
    text = text.replace('_', '-')  # Convert underscores to hyphens
    
    # Handle multiple spaces
    text = ' '.join(text.split())
    
    # Accept ANY text with reasonable length
    if len(text) >= 2:  # Minimum 2 characters
        # Check if it looks like a plate (has alphanumeric content)
        has_alnum = any(c.isalnum() for c in text)
        
        if has_alnum:
            return text
    
    # Even more relaxed - accept single characters if they're alphanumeric
    if len(text) == 1 and text.isalnum():
        return text
    
    return ""

def process_two_line_plate(plate_img, ocr_engine):
    """Enhanced processing for two-line license plates"""
    results = []
    
    # Method 1: Read entire plate first
    try:
        full_result = ocr_engine.readtext(plate_img, detail=1, paragraph=False)
        if full_result:
            results.extend(full_result)
    except:
        pass
    
    # Method 2: Split horizontally and read each half
    try:
        h, w = plate_img.shape[:2]
        
        # Split into top and bottom halves
        top_half = plate_img[:h//2, :]
        bottom_half = plate_img[h//2:, :]
        
        # Add some overlap to catch text on the border
        overlap = max(5, h//10)
        top_extended = plate_img[:h//2 + overlap, :]
        bottom_extended = plate_img[h//2 - overlap:, :]
        
        # Read each section
        for section, name in [(top_half, "top"), (bottom_half, "bottom"), 
                             (top_extended, "top_ext"), (bottom_extended, "bottom_ext")]:
            if section.shape[0] > 10 and section.shape[1] > 20:  # Minimum size check
                section_results = ocr_engine.readtext(section, detail=1, paragraph=False)
                if section_results:
                    results.extend(section_results)
    except:
        pass
    
    # Method 3: Try with different preprocessing for two-line plates
    try:
        # Enhance contrast for better line separation
        enhanced = cv2.convertScaleAbs(plate_img, alpha=1.5, beta=10)
        enhanced_results = ocr_engine.readtext(enhanced, detail=1, paragraph=False)
        if enhanced_results:
            results.extend(enhanced_results)
    except:
        pass
    
    return results

def combine_partial_results(results):
    """Enhanced combination of partial OCR results for two-line plates"""
    if not results:
        return ""
    
    # Extract text fragments with positions
    fragments = []
    for result in results:
        if len(result) >= 3:
            bbox = result[0]
            text = result[1].strip().upper()
            conf = result[2]
            
            if len(text) >= 1 and conf > 0.3:  # Lower threshold for partial text
                # Calculate center Y position for sorting
                center_y = sum([point[1] for point in bbox]) / len(bbox)
                fragments.append((text, conf, center_y, bbox))
    
    if not fragments:
        return ""
    
    # Remove duplicates (same text with similar positions)
    unique_fragments = []
    for frag in fragments:
        is_duplicate = False
        for existing in unique_fragments:
            if (frag[0] == existing[0] and 
                abs(frag[2] - existing[2]) < 20):  # Similar Y position
                if frag[1] > existing[1]:  # Keep higher confidence
                    unique_fragments.remove(existing)
                    unique_fragments.append(frag)
                is_duplicate = True
                break
        if not is_duplicate:
            unique_fragments.append(frag)
    
    # Sort by Y position (top to bottom)
    unique_fragments.sort(key=lambda x: x[2])
    
    # Group into lines based on Y position
    lines = []
    current_line = []
    last_y = -1
    
    for frag in unique_fragments:
        text, conf, y, bbox = frag
        
        if last_y == -1 or abs(y - last_y) < 15:  # Same line
            current_line.append((text, conf))
        else:  # New line
            if current_line:
                lines.append(current_line)
            current_line = [(text, conf)]
        last_y = y
    
    if current_line:
        lines.append(current_line)
    
    # Combine text from each line
    combined_lines = []
    for line in lines:
        # Sort fragments in line by confidence
        line.sort(key=lambda x: x[1], reverse=True)
        line_text = " ".join([frag[0] for frag in line])
        combined_lines.append(line_text.strip())
    
    # Join lines with space or newline
    if len(combined_lines) == 1:
        return combined_lines[0]
    elif len(combined_lines) == 2:
        # For two-line plates, join with space
        return f"{combined_lines[0]} {combined_lines[1]}"
    else:
        # Multiple fragments, join best ones
        return " ".join(combined_lines[:2])  # Take top 2 lines

def try_multiple_ocr_methods(plate_img, ocr_engine):
    """Comprehensive OCR with multiple enhancement techniques"""
    all_results = []
    
    # Get multiple enhanced versions
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()
    
    # Create multiple enhanced versions
    enhanced_versions = []
    
    # Original
    enhanced_versions.append(gray)
    
    # Upscaled versions
    for scale in [2, 3, 4]:
        h, w = gray.shape
        upscaled = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        enhanced_versions.append(upscaled)
    
    # CLAHE enhanced
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_versions.append(clahe.apply(gray))
    
    # Gamma corrections
    for gamma in [0.5, 1.5, 2.0]:
        gamma_corrected = np.array(255 * (gray / 255) ** gamma, dtype='uint8')
        enhanced_versions.append(gamma_corrected)
    
    # Threshold versions
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_versions.append(thresh_otsu)
    
    _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    enhanced_versions.append(thresh_inv)
    
    # Bilateral filter + sharpening
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(bilateral, -1, kernel)
    enhanced_versions.append(sharpened)
    
    # Try OCR on each version
    for i, enhanced in enumerate(enhanced_versions):
        try:
            # Method 1: Standard OCR
            results = ocr_engine.readtext(enhanced, detail=1, paragraph=False)
            if results:
                all_results.extend([(r[1], r[2], f"method_{i}_std") for r in results])
            
            # Method 2: Paragraph mode
            results_para = ocr_engine.readtext(enhanced, detail=1, paragraph=True)
            if results_para:
                all_results.extend([(r[1], r[2], f"method_{i}_para") for r in results_para])
            
            # Method 3: Different allowlist
            results_alpha = ocr_engine.readtext(enhanced, detail=1, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- ')
            if results_alpha:
                all_results.extend([(r[1], r[2], f"method_{i}_alpha") for r in results_alpha])
                
        except Exception as e:
            continue
    
    # Process results
    if all_results:
        # Sort by confidence
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Try to find best result
        for text, conf, method in all_results:
            cleaned_text = text.strip().upper()
            if len(cleaned_text) >= 2:  # Minimum length
                # Basic filtering
                cleaned_text = re.sub(r'[^A-Z0-9\s-]', '', cleaned_text)
                cleaned_text = ' '.join(cleaned_text.split())
                
                if len(cleaned_text) >= 2:
                    return (cleaned_text, conf)
        
        # If no good result, return best raw result
        if all_results:
            best = all_results[0]
            return (best[0].strip(), best[1])
    
    return None

def process_image_enhanced_ocr(image_path, output_dir="output"):
    """Enhanced image processing with better OCR"""
    print(f"🖼️  Processing image: {image_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Could not load image: {image_path}")
        return None

    print(f"📏 Image size: {frame.shape[1]}x{frame.shape[0]}")

    # Load models
    try:
        vehicle_model = YOLO("yolov8n.pt")
        plate_model = YOLO("license_plate_detector.pt")
        ocr_engine = easyocr.Reader(['en'])
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return None

    # Vehicle detection
    print("🔍 Detecting vehicles...")
    vehicle_results = vehicle_model(frame, verbose=False, conf=0.4)[0]
    vehicle_detections = []

    for box in vehicle_results.boxes:
        vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        vehicle_type = vehicle_model.names[cls_id]
        vehicle_detections.append([vx1, vy1, vx2, vy2, conf, vehicle_type])

    print(f"🚗 Detected {len(vehicle_detections)} vehicles")

    # Process each vehicle
    results = []
    for i, (vx1, vy1, vx2, vy2, vehicle_conf, vehicle_type) in enumerate(vehicle_detections):
        print(f"\n🔍 Processing vehicle {i+1}/{len(vehicle_detections)}: {vehicle_type}")

        # Crop vehicle
        vehicle_crop = frame[vy1:vy2, vx1:vx2]
        if vehicle_crop.size == 0:
            continue

        # Save vehicle crop
        vehicle_path = os.path.join(output_dir, f"vehicle_{i+1}_{vehicle_type}.jpg")
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
                        plate_path = os.path.join(output_dir, f"plate_{i+1}_original.jpg")
                        cv2.imwrite(plate_path, plate_crop)

                        # Try multiple OCR methods
                        best_ocr_result = try_multiple_ocr_methods(plate_crop, ocr_engine)

                        if best_ocr_result:
                            raw_text = best_ocr_result[0]
                            ocr_conf = best_ocr_result[1]

                            print(f"   Raw OCR: '{raw_text}' (confidence: {ocr_conf:.2f})")

                            # Filter text
                            filtered_text = filter_plate_text_relaxed(raw_text)

                            if filtered_text:
                                plate_text = filtered_text
                                print(f"   ✅ Filtered: '{plate_text}'")

                                # Save enhanced plate
                                enhanced_plate = enhance_plate_image_advanced(plate_crop)
                                enhanced_path = os.path.join(output_dir, f"plate_{i+1}_enhanced.jpg")
                                cv2.imwrite(enhanced_path, enhanced_plate)
                            else:
                                print(f"   ❌ Text filtered out: '{raw_text}'")
                        else:
                            print(f"   ❌ No OCR result")

        except Exception as e:
            print(f"❌ Plate detection error: {e}")

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
            'Vehicle_Image': vehicle_path,
            'Plate_Image': os.path.join(output_dir, f"plate_{i+1}_original.jpg") if plate_bbox else None
        }
        results.append(result)

        print(f"📋 Vehicle {i+1}: {vehicle_type} | Plate: {plate_text if plate_text else 'Not detected'}")

    # Create visualization
    print("\n🎨 Creating visualization...")
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

    # Print summary
    print(f"\n✅ Processing completed!")
    print(f"📊 Results saved to: {output_dir}")
    print(f"📄 CSV: {csv_path}")
    print(f"🖼️  Visualization: {vis_path}")

    if results:
        print(f"\n🎯 OCR Results:")
        for result in results:
            print(f"   Vehicle {result['Vehicle_ID']}:")
            print(f"     Raw OCR: '{result['Raw_OCR_Text']}' (conf: {result['OCR_Confidence']:.2f})")
            print(f"     Filtered: '{result['Plate_Text']}'")

    return results, df



