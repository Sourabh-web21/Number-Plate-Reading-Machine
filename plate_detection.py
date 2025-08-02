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
    """Gentle plate image enhancement for better OCR"""
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
        scale_factor = max(2, 200 // min(height, width))
        gray = cv2.resize(gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)

    # Mild CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light sharpening (reduced strength)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened

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
    """Enhanced filtering for Indian license plates"""
    if not text:
        return ""
    
    # Clean text
    text = text.strip().upper()
    
    # Remove problematic OCR artifacts but keep hyphens and spaces
    text = re.sub(r'[^\w\s-]', '', text)
    text = text.replace('_', '-')  # Convert underscores to hyphens
    
    # Handle multiple spaces
    text = ' '.join(text.split())
    
    # Split into words and filter
    words = text.split()
    filtered_words = []
    
    for word in words:
        # Skip common OCR artifacts
        if word in ['IND', 'INDIA', 'BHARAT', 'GOV', 'GOVT']:
            continue
        
        # Keep words that look like plate components
        if len(word) >= 2:
            # Check if it's alphanumeric (letters + numbers)
            has_letter = any(c.isalpha() for c in word)
            has_number = any(c.isdigit() for c in word)
            
            # Accept if it has both letters and numbers, or is reasonably long
            if (has_letter and has_number) or len(word) >= 4:
                filtered_words.append(word)
            # Accept pure letter sequences (state codes)
            elif has_letter and not has_number and len(word) <= 3:
                filtered_words.append(word)
            # Accept pure number sequences
            elif has_number and not has_letter and len(word) >= 2:
                filtered_words.append(word)
    
    # Join filtered words
    result = ' '.join(filtered_words)
    
    # Final validation - should have at least one letter and one number
    if result and (any(c.isalpha() for c in result) and any(c.isdigit() for c in result)):
        return result
    
    return ""

def process_two_line_plate(plate_img, ocr_engine):
    """Process two-line license plates"""
    results = []

    # Try reading the entire plate
    try:
        full_result = ocr_engine.readtext(plate_img, detail=1)
        if full_result:
            results.extend(full_result)
    except:
        pass

    # Try reading top and bottom separately
    try:
        h, w = plate_img.shape[:2]
        top_half = plate_img[:h//2, :]
        bottom_half = plate_img[h//2:, :]

        top_result = ocr_engine.readtext(top_half, detail=1)
        bottom_result = ocr_engine.readtext(bottom_half, detail=1)

        if top_result:
            results.extend(top_result)
        if bottom_result:
            results.extend(bottom_result)
    except:
        pass

    return results

def combine_partial_results(results):
    """Combine partial OCR results into complete plate"""
    if not results:
        return ""

    # Extract all text fragments
    texts = []
    for result in results:
        if len(result) >= 3:
            text = result[1].strip().upper()
            conf = result[2]
            if len(text) >= 2:
                texts.append((text, conf))

    if not texts:
        return ""

    # Sort by confidence
    texts.sort(key=lambda x: x[1], reverse=True)

    # Try to combine fragments
    combined = ""
    for text, conf in texts:
        if not combined:
            combined = text
        else:
            # Check if this fragment adds new information
            if not any(frag in combined for frag in text.split()):
                combined += " " + text

    return combined.strip()

def try_multiple_ocr_methods(plate_img, ocr_engine):
    """Try multiple OCR approaches and return the best result"""
    results = []

    # Method 1: Process as two-line plate
    try:
        two_line_results = process_two_line_plate(plate_img, ocr_engine)
        results.extend(two_line_results)
    except:
        pass

    # Method 2: Enhanced image
    try:
        enhanced = enhance_plate_image_advanced(plate_img)
        enhanced_results = process_two_line_plate(enhanced, ocr_engine)
        results.extend(enhanced_results)
    except:
        pass

    # Method 3: Upscaled
    try:
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img
        upscaled = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2), interpolation=cv2.INTER_CUBIC)
        upscaled_results = process_two_line_plate(upscaled, ocr_engine)
        results.extend(upscaled_results)
    except:
        pass

    # Method 4: Original image
    try:
        ocr1 = ocr_engine.readtext(plate_img, detail=1)
        if ocr1:
            results.extend(ocr1)
    except:
        pass

    # Method 5: Grayscale
    try:
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img
        ocr3 = ocr_engine.readtext(gray, detail=1)
        if ocr3:
            results.extend(ocr3)
    except:
        pass

    # Get best single result instead of combining
    if results:
        # Find the best result by confidence
        best_result = max(results, key=lambda x: x[2] if len(x) >= 3 else 0)
        
        if len(best_result) >= 3:
            raw_text = best_result[1].strip().upper()
            confidence = best_result[2]
            
            # Filter the best result
            filtered_text = filter_plate_text_relaxed(raw_text)
            
            if filtered_text:
                return (filtered_text, confidence)
            else:
                # Return raw text if filtering removes everything
                return (raw_text, confidence)

    return None

def try_multiple_ocr_methods_aggressive(plate_img, ocr_engine):
    """More aggressive OCR with additional preprocessing"""
    results = []
    
    # Method 1: Original try_multiple_ocr_methods
    original_result = try_multiple_ocr_methods(plate_img, ocr_engine)
    if original_result:
        results.append(original_result)
    
    # Method 2: More aggressive enhancement
    try:
        # Stronger enhancement
        enhanced = enhance_plate_image_for_ocr(plate_img)
        
        # Try with different scales
        for scale in [1.5, 2.0, 2.5]:
            if len(enhanced.shape) == 3:
                gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            else:
                gray = enhanced
                
            h, w = gray.shape
            scaled = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
            
            ocr_results = ocr_engine.readtext(scaled, detail=1)
            if ocr_results:
                # Get best result
                best = max(ocr_results, key=lambda x: x[2] if len(x) >= 3 else 0)
                if len(best) >= 3 and best[2] > 0.3:
                    results.append((best[1], best[2]))
    except:
        pass
    
    # Return best result
    if results:
        return max(results, key=lambda x: x[1] if len(x) >= 2 else 0)
    
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








