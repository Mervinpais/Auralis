#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np
import onnxruntime as ort
import subprocess
import psutil

# ===== CONFIG =====
MODEL_PATH = "yolov5n.onnx"
FRAME_SIZE = (640, 640)
CONF_THRESHOLD = 0.5
TERMINAL_IMAGE_WIDTH = "100"
RAM_TMP_FILE = "/dev/shm/tmp_frame.png"

# COCO Class Names
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "knife", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ===== UTILS =====
def letterbox(frame, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = frame.shape[:2]
    r = min(new_shape[0]/h, new_shape[1]/w)
    new_unpad = (int(w*r), int(h*r))
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    dw, dh = dw/2, dh/2
    resized = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(dh), int(dh+0.5)
    left, right = int(dw), int(dw+0.5)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return padded

def preprocess(frame):
    padded = letterbox(frame, FRAME_SIZE)
    blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    blob = np.transpose(blob, (2,0,1))[np.newaxis,...].copy()
    return blob, padded

def postprocess(outputs, conf_thresh=0.5, iou_thresh=0.4):
    boxes, scores, class_ids = [], [], []
    output = outputs[0]
    for det in output[0]:
        conf = float(det[4])
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = det[:4]
        cls_id = int(det[5])
        boxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
        scores.append(conf)
        class_ids.append(cls_id)

    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
        indices = indices.flatten() if len(indices) > 0 else []
        boxes_nms = [[int(x), int(y), int(x+w), int(y+h)] for i in indices for x,y,w,h in [boxes[i]]]
        scores_nms = [scores[i] for i in indices]
        class_ids_nms = [class_ids[i] for i in indices]
        return boxes_nms, scores_nms, class_ids_nms
    return [], [], []

def draw_boxes(frame, boxes, scores, class_ids):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        if cls_id >= len(CLASS_NAMES):
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{CLASS_NAMES[cls_id]}:{score:.2f}", (x1,max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame

def display_frame(frame, detected_objects):
    os.system("clear")
    if detected_objects:
        print("Detected:", ", ".join(detected_objects[:3]))
    cv2.imwrite(RAM_TMP_FILE, frame)
    subprocess.run(["catimg", "-w", TERMINAL_IMAGE_WIDTH, RAM_TMP_FILE],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def available_memory_kb():
    return psutil.virtual_memory().available // 1024

# ===== LOAD MODEL =====
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    exit(1)

print("🧠 Loading model...")
sess = ort.InferenceSession(MODEL_PATH)
input_name = sess.get_inputs()[0].name
print("✅ Model loaded!")

# ===== CAMERA =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera")
    exit(1)

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        if available_memory_kb() < 100_000:
            print("⚠ Low memory — skipping frame")
            time.sleep(0.05)
            continue

        blob, padded_frame = preprocess(frame)
        outputs = sess.run(None, {input_name: blob})
        boxes, scores, class_ids = postprocess(outputs, CONF_THRESHOLD, 0.4)
        frame_with_boxes = draw_boxes(padded_frame, boxes, scores, class_ids)

        detected = []
        for cls_id, score in zip(class_ids, scores):
            if cls_id < len(CLASS_NAMES):
                detected.append(f"{CLASS_NAMES[cls_id]}:{score:.2f}")

        display_frame(frame_with_boxes, detected)
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n🛑 Exiting...")
finally:
    cap.release()

