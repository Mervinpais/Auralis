import cv2
import easyocr
import asyncio
import subprocess
import hashlib
import os
from datetime import datetime

# --- Voice Cache Setup ---
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
        print(f"[ERR] Piper TTS error: {e}")
    return output_file

async def speak(text):
    output_file = text_to_filename(text)
    if not os.path.exists(output_file):
        output_file = create_tts_file(text)
    try:
        subprocess.run(["aplay", "-D", "default", output_file], check=True)
    except Exception as e:
        print(f"[ERR] TTS playback error: {e}")

# --- EasyOCR Reader ---
reader = easyocr.Reader(['en'])

# --- Capture frame from camera ---
def capture_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERR] Camera not found.")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERR] Failed to capture frame.")
        return None
    return frame

# --- Main loop ---
async def main():
    print("[INFO] OCR ready. Press Enter to scan text (Ctrl+C to exit).")
    await speak("OCR ready. Press enter to scan text.")

    while True:
        try:
            input("Press Enter to scan...")
            frame = capture_frame()
            if frame is None:
                continue

            # EasyOCR expects RGB images
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = reader.readtext(rgb_frame)

            if not results:
                print("No text detected.")
                await speak("No text detected")
                continue

            detected_text = " ".join([res[1] for res in results])
            print(f"[OCR] {detected_text}")
            await speak(detected_text)

        except KeyboardInterrupt:
            print("Exiting...")
            break

# --- Run ---
asyncio.run(main())

