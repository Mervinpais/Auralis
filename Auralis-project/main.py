#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np
import onnxruntime as ort
import subprocess
import tempfile
import psutil

# ===== CONFIG =====
MODEL_PATH = "yolov5n.onnx"
CONF_THRESHOLD = 0.5
PRIORITY_SET = {"person", "bottle", "book"}  # items to report
FRAME_SIZE = (640, 640)                        # input size
SAVE_INTERVAL = 5                              # seconds
SAVE_DIR = "./snapshots"
TERMINAL_IMAGE_WIDTH = "100"                   # for catimg

os.makedirs(SAVE_DIR, exist_ok=True)

# ===== UTILS =====
def letterbox(frame, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize image to fit into new_shape, preserving aspect ratio with padding."""
    shape = frame.shape[:2]  # current shape: height, width
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(shape[1] * r), int(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    frame_resized = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(dh), int(dh + 0.5)
    left, right = int(dw), int(dw + 0.5)
    frame_padded = cv2.copyMakeBorder(frame_resized, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=color)
    return frame_padded, r, (dw, dh)

def preprocess(frame):
    frame_padded, scale, pad = letterbox(frame, FRAME_SIZE)
    blob = cv2.cvtColor(frame_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)
    return blob, frame_padded, scale, pad

def postprocess(outputs, conf_thresh=0.5, iou_thresh=0.4):
    """Apply confidence thresholding and NMS on ONNX YOLO outputs."""
    boxes, scores, class_ids = [], [], []
    output = outputs[0]  # shape: [1, num_boxes, 6] (x1,y1,x2,y2,conf,class)
    
    for det in output[0]:
        conf = float(det[4])
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = det[:4]
        cls_id = int(det[5])
        boxes.append([x1, y1, x2 - x1, y2 - y1])  # convert to [x,y,w,h] for NMS
        scores.append(conf)
        class_ids.append(cls_id)
    
    # Apply OpenCV NMS
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
        if len(indices) > 0:
            # flatten indices safely
            if isinstance(indices[0], (list, tuple, np.ndarray)):
                indices = [i[0] for i in indices]
            else:
                indices = [int(i) for i in indices]

            boxes_nms = [boxes[i] for i in indices]
            scores_nms = [scores[i] for i in indices]
            class_ids_nms = [class_ids[i] for i in indices]
    
            # convert back to x1,y1,x2,y2
            boxes_nms = [[int(x), int(y), int(x + w), int(y + h)] for x, y, w, h in boxes_nms]
            return boxes_nms, scores_nms, class_ids_nms
    return [], [], []



    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
        boxes_nms = [boxes[i[0]] for i in indices]
        scores_nms = [scores[i[0]] for i in indices]
        class_ids_nms = [class_ids[i[0]] for i in indices]
        # Convert back to x1,y1,x2,y2
        boxes_nms = [[int(x), int(y), int(x + w), int(y + h)] for x, y, w, h in boxes_nms]
        return boxes_nms, scores_nms, class_ids_nms
    else:
        return [], [], []

def draw_detections(frame, boxes, scores, class_ids, class_names):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        if cls_id >= len(class_names):
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        label = f"{class_names[cls_id]}: {score:.2f}"
        cv2.putText(frame, label, (x1, max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame

def display_frame_terminal(frame):
    # Save to temp file and show in terminal with catimg
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        cv2.imwrite(tmp.name, frame)
        subprocess.run(["catimg", "-w", TERMINAL_IMAGE_WIDTH, tmp.name])

def get_available_kb():
    return psutil.virtual_memory().available // 1024

# ===== LOAD MODEL =====
if not os.path.exists(MODEL_PATH):
    print(f"❌ ONNX model not found at {MODEL_PATH}")
    exit(1)

print("🧠 Loading ONNX model...")
sess = ort.InferenceSession(MODEL_PATH)
input_name = sess.get_inputs()[0].name
print("✅ Model loaded!")

# Dummy class names (replace with actual YOLOv5 names if different)
class_names = ["person", "bottle", "book", "cellphone", "chair"]

# ===== CAMERA =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera")
    exit(1)

last_save_time = time.time()
print("📷 Headless camera feed running. Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        # Skip inference if low memory
        if get_available_kb() < 100000:  # ~100MB
            print("⚠️ Low memory — skipping this frame")
            time.sleep(0.05)
            continue

        blob, frame_resized, scale, pad = preprocess(frame)
        outputs = sess.run(None, {input_name: blob.astype(np.float32)})

        boxes, scores, class_ids = postprocess(outputs, CONF_THRESHOLD, 0.4)
        frame_with_boxes = draw_detections(frame_resized, boxes, scores, class_ids, class_names)

        # Priority detection
        found = {}
        for score, cls_id in zip(scores, class_ids):
            label = class_names[cls_id]
            if label in PRIORITY_SET:
                if label not in found or score > found[label]:
                    found[label] = score

        if found:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            items = ", ".join(f"{k}({v:.2f})" for k, v in found.items())
            print(f"[{ts}] Detected priorities: {items}")

        # Terminal display
        display_frame_terminal(frame_with_boxes)

        # Save snapshot every SAVE_INTERVAL seconds
        if time.time() - last_save_time >= SAVE_INTERVAL:
            ts_filename = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(SAVE_DIR, f"detection_{ts_filename}.png")
            cv2.imwrite(save_path, frame_with_boxes)
            print(f"[INFO] Saved snapshot: {save_path}")
            last_save_time = time.time()

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n🛑 Exiting...")
finally:
    cap.release()

