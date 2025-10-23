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

# --- Show preview with catimg ---
async def show_face_preview(face_img):
    temp_file = "tmp_preview.jpg"
    preview = cv2.resize(face_img, (50,50))
    cv2.imwrite(temp_file, preview)
    await asyncio.to_thread(subprocess.run, ["catimg", temp_file])
    os.remove(temp_file)

# --- Save new face ---
async def maybe_save_face(face_img, encoding):
    await show_face_preview(face_img)
    ans = input("Save this face? (yes/no): ").strip().lower()
    if ans != "yes":
        print("Skipped saving face.")
        return None, None

    name = input("Enter a name for this person: ").strip()
    if not name:
        print("No name entered, skipping.")
        return None, None

    timestamp = datetime.now().strftime("%Y_%m_%d.%H_%M_%S")
    face_file = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    npy_file = os.path.join(EMBEDDINGS_DIR, f"{name}.npy")
    cv2.imwrite(face_file, face_img)
    np.save(npy_file, encoding)
    print(f"Saved new face: {name}")
    await speak(f"Saved new face at {timestamp}")
    return encoding, name

# --- Main async function ---
async def main():
    known_encodings, known_names = load_known_faces()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not found.")

    print("Face recognition started. Press Ctrl+C to stop.")

    try:
        while True:
            # Grab a frame only when we detect faces
            ret, frame = cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                face_img = frame[top:bottom, left:right]

                name = "Unknown"
                if known_encodings:
                    matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
                    face_distances = face_recognition.face_distance(known_encodings, encoding)
                    best_match_idx = np.argmin(face_distances)
                    if matches[best_match_idx]:
                        name = known_names[best_match_idx]

                # Print and async speak after 0.5s
                print(f"Detected: {name}")
                await asyncio.sleep(0.5)
                await speak(f"Detected {name}")

                # Save unknown faces
                if name == "Unknown":
                    emb, n = await maybe_save_face(face_img, encoding)
                    if emb is not None:
                        known_encodings.append(emb)
                        known_names.append(n)

    except KeyboardInterrupt:
        print("\nStopping face recognition.")
    finally:
        cap.release()

# --- Run async main ---
asyncio.run(main())

