import cv2
import os
import numpy as np
from deepface import DeepFace
import asyncio
import subprocess
import shutil
from datetime import datetime
import hashlib
import time

# --- Directories ---
BASE_DIR = "faces"
KNOWN_DIR = os.path.join(BASE_DIR, "known")
THREAT_DIR = os.path.join(BASE_DIR, "threat")
os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(THREAT_DIR, exist_ok=True)

# --- Voice Cache ---
VOICE_CACHE_DIR = "voice_cache"
os.makedirs(VOICE_CACHE_DIR, exist_ok=True)

def text_to_filename(text):
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return os.path.join(VOICE_CACHE_DIR, f"{h}.wav")

def create_tts_file(text):
    output_file = text_to_filename(text)
    if os.path.exists(output_file):
        return output_file
    try:
        subprocess.run([
            "piper",
            "--model", "en_US-amy-medium.onnx",
            "--output_file", output_file
        ], input=text.encode("utf-8"), check=True)
    except Exception as e:
        print(f"TTS error: {e}")
    return output_file

async def speak(text):
    output_file = text_to_filename(text)
    if not os.path.exists(output_file):
        output_file = create_tts_file(text)
    try:
        subprocess.run(["aplay", "-D", "default", output_file], check=True)
    except Exception as e:
        print(f"TTS playback error: {e}")

# --- Load known faces and compute mean embeddings ---
def get_face_db(folder):
    db = {}
    for name in os.listdir(folder):
        person_dir = os.path.join(folder, name)
        if not os.path.isdir(person_dir):
            continue
        embeddings = []
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            try:
                emb = DeepFace.represent(
                    img_path=img_path,
                    model_name="ArcFace",
                    enforce_detection=False
                )[0]["embedding"]
                emb = np.array(emb)
                emb = emb / np.linalg.norm(emb)  # normalize
                embeddings.append(emb)
            except:
                continue
        if embeddings:
            db[name] = np.mean(embeddings, axis=0)
    return db

# --- Compare face using cosine similarity ---
def compare_face(embedding, db, threshold=0.5):
    embedding = embedding / np.linalg.norm(embedding)
    best_name, best_score = None, -1
    for name, ref_emb in db.items():
        ref_emb = ref_emb / np.linalg.norm(ref_emb)
        score = np.dot(embedding, ref_emb)
        if score > threshold and score > best_score:
            best_score, best_name = score, name
    return best_name

# --- Capture a frame from camera ---
def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# --- Save a cropped face ---
def save_face(name, frame, folder):
    try:
        faces = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
    except:
        return "no_face"
    if len(faces) == 0:
        return "no_face"
    elif len(faces) > 1:
        return "too_many"

    face_img = faces[0]["face"]
    if face_img.max() <= 1:
        face_img = (face_img * 255).astype(np.uint8)
    else:
        face_img = face_img.astype(np.uint8)

    save_dir = os.path.join(folder, name)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(file_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
    return "saved"

# --- Main loop ---
async def main():
    known_db = get_face_db(KNOWN_DIR)
    threat_db = get_face_db(THREAT_DIR)

    await speak("Face recognition started")
    print("[INFO] Auralis ready. Type commands or let it auto-scan...")

    while True:
        cmd = input("> ").strip().lower()

        if cmd == "exit":
            await speak("Goodbye")
            break

        elif cmd == "save face":
            frame = capture_frame()
            if frame is None:
                await speak("Camera error")
                continue

            result = save_face("temp", frame, KNOWN_DIR)
            if result == "no_face":
                await speak("No face detected")
            elif result == "too_many":
                await speak("Too many faces")
            else:
                await speak("Face detected, name?")
                name = input("Enter name: ").strip()
                os.rename(os.path.join(KNOWN_DIR, "temp"), os.path.join(KNOWN_DIR, name))
                await speak(f"Name saved as {name}")
                known_db = get_face_db(KNOWN_DIR)

        elif cmd.startswith("shift "):
            name = cmd.replace("shift ", "").strip()
            src = os.path.join(KNOWN_DIR, name)
            dest = os.path.join(THREAT_DIR, name)
            if os.path.exists(src):
                shutil.move(src, dest)
                await speak(f"Shifted {name} to threat")
                known_db = get_face_db(KNOWN_DIR)
                threat_db = get_face_db(THREAT_DIR)
            else:
                await speak(f"{name} not found in known")

        elif cmd == "scan":
            frame = capture_frame()
            time.sleep(0.5)
            if frame is None:
                continue
            try:
                faces = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
                for face in faces:
                    face_img = face["face"]
                    if face_img.max() <= 1:
                        face_img = (face_img * 255).astype(np.uint8)
                    else:
                        face_img = face_img.astype(np.uint8)
                    emb = DeepFace.represent(face_img, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
                    emb = np.array(emb)
                    emb = emb / np.linalg.norm(emb)

                    threat_match = compare_face(emb, threat_db)
                    if threat_match:
                        await speak(f"Threat {threat_match}")
                        continue

                    known_match = compare_face(emb, known_db)
                    if known_match:
                        print(known_match)
                        await speak(known_match)
            except Exception as e:
                print(f"[WARN] Scan error: {e}")

        else:
            await speak("Unknown command")

# --- Run ---
asyncio.run(main())



















































































































































































