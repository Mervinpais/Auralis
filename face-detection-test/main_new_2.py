import cv2
import numpy as np
import os
import time
from datetime import datetime
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
def speak(text):
    try:
        subprocess.run(["espeak", "-ven+m3", "-s140", text])
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
# --- Main loop ---
known_encodings, known_names = load_known_faces()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not found.")

print("Face recognition started. Running headless.")

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
            if (name !=  "Unknown"):
                print(f"Detected: {name}, confidence: {confidence:.2f}")

            # New face detected
            if name == "Unknown" and False:
                speak("New face detected. Please enter the name:")
#                name_input = input("Enter name for this person: ").strip()
                face_samples = [face_img]

                # Capture 2 more frames for accuracy
                count = 1
                while count < 3:
                    ret, f = cap.read()
                    if not ret:
                        continue
                    h2, w2 = f.shape[:2]
                    blob2 = cv2.dnn.blobFromImage(f, 1.0, (300,300), (104,177,123), swapRB=False)
                    face_net.setInput(blob2)
                    detections2 = face_net.forward()
                    for j in range(detections2.shape[2]):
                        conf2 = detections2[0,0,j,2]
                        if conf2 < 0.5:
                            continue
                        box2 = (detections2[0,0,j,3:7] * np.array([w2,h2,w2,h2])).astype(int)
                        x1_, y1_, x2_, y2_ = box2
                        new_face = f[y1_:y2_, x1_:x2_]
                        if new_face.size == 0 or new_face.shape[0] < 20 or new_face.shape[1] < 20:
                            continue
                        face_samples.append(new_face)
                        count += 1
                        break
                    time.sleep(0.5)

               # embeddings = save_new_face(face_samples, name_input)
               # known_encodings.extend(embeddings)
               # known_names.extend([name_input]*3)

except KeyboardInterrupt:
    print("\nStopping face recognition.")
finally:
    cap.release()

