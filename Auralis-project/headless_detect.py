import cv2
import subprocess
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov5s.pt")  # or yolov8n.pt

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not found or can't be opened.")

last_spoken = {}  # label -> timestamp of last time it was spoken
COOLDOWN = 5      # seconds before it can be spoken again

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model.predict(source=frame, imgsz=384, device="cpu", verbose=False)

    # Collect detections
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            if conf >= 0.5:
                label = model.names[cls_id]
                detections.append((label, conf))

    # Sort by confidence, keep top 3
    detections = sorted(detections, key=lambda x: x[1], reverse=True)[:3]

    # Speak only new detections not spoken recently
    now = time.time()
    for label, conf in detections:
        last_time = last_spoken.get(label, 0)
        if now - last_time > COOLDOWN:
            print(f"Detected: {label} ({conf:.2f})")
            subprocess.run(["espeak-ng", f"I see a {label}"])
            last_spoken[label] = now

cap.release()

