# the code is originally designed to run in google colab environment



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

# === Initial Setup ===
if not os.path.exists("segment-anything"):
    subprocess.run(["git", "clone", "https://github.com/facebookresearch/segment-anything.git"])
if not os.path.exists("sam_vit_b.pth"):
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "sam_vit_b.pth"
    )

import sys
sys.path.append("segment-anything")
from segment_anything import SamPredictor, sam_model_registry

if not os.path.exists("sort"):
    subprocess.run(["git", "clone", "https://github.com/abewley/sort.git"])
import matplotlib
matplotlib.use('Agg')
from sort.sort import Sort

from google.colab.patches import cv2_imshow  # For Colab display

# === Load Models ===
plate_model = YOLO("license_plate_detector.pt")
vehicle_model = YOLO("100epoch_best.pt")
ocr_engine = easyocr.Reader(['en'])

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
sam.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
predictor = SamPredictor(sam)

# === Video Setup ===
cap = cv2.VideoCapture("/content/10_vehicle_videos.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter("tracked_output.avi", cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

plate_tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3)
vehicle_tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3)

red_line_y = frame_height // 3
tolerance = 50
frame_num = 0
SIM_THRESHOLD = 0.85

ocr_results = {}  # track_id: (plate_text, plate_conf, frame_num, vehicle_type, vehicle_id, vehicle_conf)
unique_texts = []
vehicle_logged = set()

def deskew(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    if abs(moments["mu02"]) < 1e-2:
        return img
    skew = moments["mu11"] / moments["mu02"]
    M = np.float32([[1, skew, -0.5 * skew * img.shape[0]], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

def is_similar_plate(new, existing_list, threshold=SIM_THRESHOLD):
    return any(difflib.SequenceMatcher(None, new, old).ratio() >= threshold for old in existing_list)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        vehicle_results = vehicle_model(frame, verbose=False)[0]
        vehicle_detections = []
        for box in vehicle_results.boxes:
            vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            vehicle_detections.append([vx1, vy1, vx2, vy2, conf])

        vehicle_np = np.array(vehicle_detections) if vehicle_detections else np.empty((0, 5))
        tracked_vehicles = vehicle_tracker.update(vehicle_np)

        plate_detections = []
        vehicle_info_map = {}
        for vehicle in tracked_vehicles:
            vx1, vy1, vx2, vy2, vehicle_id = map(int, vehicle)
            vehicle_crop = frame[vy1:vy2, vx1:vx2]
            vehicle_type = "Unknown"
            vehicle_conf = 0.0
            if vehicle_crop.size > 0:
                try:
                    cls_results = vehicle_model(vehicle_crop, verbose=True)[0]
                    if cls_results.boxes:
                        cls_id = int(cls_results.boxes[0].cls[0])
                        vehicle_type = vehicle_model.names[cls_id]
                        vehicle_conf = float(cls_results.boxes[0].conf[0])
                except Exception as e:
                    print(f"[Vehicle Detection Error] Frame {frame_num}, Vehicle ID {vehicle_id}: {e}")

            plate_result = None
            if vehicle_crop.size > 0:
                try:
                    plate_result = plate_model(vehicle_crop, verbose=True)[0]
                except Exception as e:
                    print(f"[Plate Detection Error] Frame {frame_num}, Vehicle ID {vehicle_id}: {e}")
            if plate_result is not None and plate_result.boxes:
                for box in plate_result.boxes:
                    px1, py1, px2, py2 = map(int, box.xyxy[0])
                plate_abs = (vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2)
                plate_detections.append((plate_abs, vehicle_id))
                vehicle_info_map[vehicle_id] = (vehicle_type, (vx1, vy1, vx2, vy2), vehicle_conf)

        detections = [[x1, y1, x2, y2, 0.99] for ((x1, y1, x2, y2), _) in plate_detections]
        dets_np = np.array(detections) if detections else np.empty((0, 5))
        tracks = plate_tracker.update(dets_np)

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            center_y = (y1 + y2) // 2
            if abs(center_y - red_line_y) > tolerance:
                continue

            predictor.set_image(frame)
            input_box = np.array([x1, y1, x2, y2])
            masks, _, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
            mask = masks[0]
            masked = cv2.bitwise_and(frame, frame, mask=mask.astype(np.uint8))
            cropped = masked[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
            limg = cv2.merge((cl, a, b))
            clahe_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            table = np.array([(i / 255.0) ** (1 / 1.5) * 255 for i in range(256)]).astype("uint8")
            gamma_img = cv2.LUT(clahe_img, table)
            upscale = cv2.resize(gamma_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            denoised = cv2.fastNlMeansDenoisingColored(upscale, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
            deskewed = deskew(denoised)
            sharpened = cv2.filter2D(deskewed, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
            gray_final = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

            try:
                result = ocr_engine.readtext(gray_final, detail=1)
                if result:
                    for line in result:
                        text = line[1].strip().upper().replace(" ", "")
                        conf = float(line[2])
                        if len(text) >= 5 and not is_similar_plate(text, unique_texts):
                            matched_vehicle_id = None
                            for ((px1, py1, px2, py2), v_id) in plate_detections:
                                if abs(px1 - x1) < 10 and abs(py1 - y1) < 10:
                                    matched_vehicle_id = v_id
                                    break
                            if matched_vehicle_id is not None and matched_vehicle_id not in vehicle_logged:
                                vehicle_type, (vx1, vy1, vx2, vy2), vehicle_conf = vehicle_info_map[matched_vehicle_id]
                                ocr_results[track_id] = (text, conf, frame_num, vehicle_type, matched_vehicle_id, vehicle_conf)
                                unique_texts.append(text)
                                vehicle_logged.add(matched_vehicle_id)

                                # Save and show vehicle image in Colab
                                vehicle_img = frame[vy1:vy2, vx1:vx2]
                                img_path = f"/content/vehicle_{matched_vehicle_id}_frame{frame_num}.jpg"
                                cv2.imwrite(img_path, vehicle_img)
                                print(f"📸 Vehicle ID {matched_vehicle_id} | Plate: {text} | Frame: {frame_num}")
                                cv2_imshow(vehicle_img)
                            break
            except Exception as e:
                print(f"[OCR Error] Frame {frame_num}: {e}")
                continue

            label = ""
            if track_id in ocr_results:
                plate_text = ocr_results[track_id][0]
                vehicle_type = ocr_results[track_id][3]
                label = f"{plate_text} ({vehicle_type})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {int(track_id)} {label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.line(frame, (0, red_line_y), (frame_width, red_line_y), (0, 0, 255), 2)
        out.write(frame)

except KeyboardInterrupt:
    print("🛑 Interrupted by user")

finally:
    cap.release()
    out.release()
    final_records = [
        (info[2], track_id, info[0], info[3], info[4], info[1], info[5])
        for track_id, info in ocr_results.items()
    ]
    df = pd.DataFrame(final_records, columns=["Frame", "Track_ID", "Plate", "Vehicle_Type", "Vehicle_ID", "Plate_Conf", "Vehicle_Conf"])
    df.to_csv("detected_plates.csv", index=False)
    print("✅ Saved: detected_plates.csv and tracked_output.avi")