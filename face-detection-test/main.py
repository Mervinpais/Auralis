import cv2
import numpy as np
import os
import time
from datetime import datetime
import pyttsx3
import subprocess

# --- Paths ---
KNOWN_FACES_DIR = "known_faces"
EMBEDDINGS_DIR = "known_embeddings"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# --- Load models ---
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
embedding_model = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")

# --- Recognition settings ---
THRESHOLD = 0.6

# --- Text-to-Speech setup ---
tts_engine = pyttsx3.init(driverName='espeak')
def speak(text):
    try:
        subprocess.run(["espeak", "-ven+m3", "-s140", text])
#        tts_engine.say(text)
#        tts_engine.runAndWait()
    except Exception as e:
        print(f"Speech error: {e}")

# --- Load known faces ---
def load_known_faces():
    known_encodings = []
    known_names = []
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".npy"):
            name = file[:-4]
            vec = np.load(os.path.join(EMBEDDINGS_DIR, file))
            known_encodings.append(vec)
            known_names.append(name)
    return known_encodings, known_names

# --- Compute face embedding ---
def get_face_embedding(face_img):
    face_blob = cv2.dnn.blobFromImage(face_img, 1.0/255, (96,96), (0,0,0), swapRB=True, crop=True)
    embedding_model.setInput(face_blob)
    vec = embedding_model.forward()
    return vec.flatten()

# --- Save new face with timestamp ---
def save_new_face(face_img, embedding):
    timestamp = datetime.now().strftime("%Y_%m_%d.%H_%M_%S")
    face_file = os.path.join(KNOWN_FACES_DIR, f"{timestamp}.jpg")
    npy_file = os.path.join(EMBEDDINGS_DIR, f"{timestamp}.npy")
    cv2.imwrite(face_file, face_img)
    np.save(npy_file, embedding)
    print(f"Saved new face: {timestamp}")
    speak(f"Saved new face at {timestamp}")

# --- Main loop ---
known_encodings, known_names = load_known_faces()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not found.")

print("Face recognition started. Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104,177,123), swapRB=False)
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence < 0.5:
                continue

            box = (detections[0,0,i,3:7] * np.array([w,h,w,h])).astype(int)
            x1, y1, x2, y2 = box
            face_img = frame[y1:y2, x1:x2]

            if face_img.size == 0 or face_img.shape[0] < 20 or face_img.shape[1] < 20:
                continue

            embedding = get_face_embedding(face_img)
            name = "Unknown"

            if known_encodings:
                sims = [np.dot(embedding, k)/(np.linalg.norm(embedding)*np.linalg.norm(k)) for k in known_encodings]
                best_idx = np.argmax(sims)
                if sims[best_idx] > THRESHOLD:
                    name = known_names[best_idx]

            print(f"Detected: {name}, confidence: {confidence:.2f}")
            speak(f"Detected {name}")
            # Save unknown faces automatically
            if name == "Unknown":
                save_new_face(face_img, embedding)
                known_encodings.append(embedding)
                known_names.append(datetime.now().strftime("%Y_%m_%d.%H_%M_%S"))

except KeyboardInterrupt:
    print("\nStopping face recognition.")
finally:
    cap.release()


