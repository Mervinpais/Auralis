import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
import asyncio
import subprocess

# --- Paths ---
KNOWN_FACES_DIR = "known_faces"
EMBEDDINGS_DIR = "known_embeddings"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# --- Async Text-to-Speech ---
async def speak(text):
    await asyncio.to_thread(subprocess.run, ["espeak", "-ven+m3", "-s140", text])

# --- Helper: Average encodings per person ---
def get_mean_encoding(folder):
    encodings = []
    for img in os.listdir(folder):
        path = os.path.join(folder, img)
        if not path.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        image = face_recognition.load_image_file(path)
        faces = face_recognition.face_encodings(image)
        if faces:
            encodings.append(faces[0])
    if not encodings:
        return None
    return np.mean(encodings, axis=0)

# --- Load known faces ---
def load_known_faces():
    known_encodings = []
    known_names = []

    for folder in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, folder)
        if os.path.isdir(person_dir):
            mean_enc = get_mean_encoding(person_dir)
            if mean_enc is not None:
                known_encodings.append(mean_enc)
                known_names.append(folder)
    return known_encodings, known_names

# --- Save new face set ---
async def save_new_face(name, face_img, encoding):
    folder = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(folder, f"{timestamp}.jpg"), face_img)
    print(f"Saved new image for {name}")

# --- Main async function ---
async def main():
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

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                face_img = frame[top:bottom, left:right]
                name = "Unknown"

                if known_encodings:
                    face_distances = face_recognition.face_distance(known_encodings, encoding)
                    best_match_idx = np.argmin(face_distances)
                    if face_distances[best_match_idx] < 0.42:  # stricter threshold
                        name = known_names[best_match_idx]

                print(f"Detected: {name}")
                await asyncio.sleep(0.5)
                await speak(f"Detected {name}")

                # Save new faces interactively
                if name == "Unknown":
                    cv2.imshow("Unknown Face", face_img)
                    cv2.waitKey(1)
                    ans = input("Save this new person? (yes/no): ").strip().lower()
                    if ans == "yes":
                        new_name = input("Enter name: ").strip()
                        if new_name:
                            await save_new_face(new_name, face_img, encoding)
                            # Reload encodings
                            known_encodings, known_names = load_known_faces()
                    cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("\nStopping face recognition.")
    finally:
        cap.release()

# --- Run async main ---
asyncio.run(main())
