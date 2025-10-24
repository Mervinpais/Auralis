import cv2
import easyocr
import pytesseract
import asyncio
import subprocess
import hashlib
import os
import tempfile
from datetime import datetime
import sys
import termios
import tty

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

async def speak(text, fast=False):
    """
    Speak the given text.

    Parameters:
        text (str): Text to speak.
        fast (bool): If True, use espeak (fast). 
                     If False, use cached Piper TTS and aplay (slower but nicer voice).
    """
    if fast:
        # Fast TTS using espeak
        try:
            subprocess.run(["espeak", "-ven+m3", "-s140", text], check=True)
        except Exception as e:
            print(f"[ERR] espeak failed: {e}")
        return

    # Slow, high-quality TTS using Piper + aplay
    output_file = text_to_filename(text)
    if not os.path.exists(output_file):
        output_file = create_tts_file(text)

    try:
        subprocess.run(["aplay", "-D", "default", output_file], check=True)
    except Exception as e:
        print(f"[ERR] TTS playback error: {e}")


# --- EasyOCR Reader ---
easy_reader = easyocr.Reader(['en'])

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

# --- Print frame in console using catimg ---
def show_frame_in_console(frame):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        cv2.imwrite(tmp_file.name, frame)
        try:
            subprocess.run(["catimg", tmp_file.name], check=True)
        except Exception as e:
            print(f"[ERR] catimg failed: {e}")
        finally:
            os.remove(tmp_file.name)

# --- Capture single keypress (headless-friendly) ---
def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# --- Main loop ---
async def main():
    print("[INFO] OCR ready. Press Enter for EasyOCR, Space for pytesseract, Ctrl+C to exit.")
    await speak("OCR ready. Press enter for easy OCR, space for tesseract OCR.")

    while True:
        try:
            key = get_key()
            if key == '\x03':  # Ctrl+C
                print("Exiting...")
                break
            elif key == '\r' or key == '\n':  # Enter
                method = "EasyOCR"
            elif key == ' ':
                method = "Tesseract"
            else:
                continue

            frame = capture_frame()
            if frame is None:
                continue

            # Show frame in console
            show_frame_in_console(frame)

            # Convert to RGB for EasyOCR, grayscale for Tesseract
            if method == "EasyOCR":
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = easy_reader.readtext(rgb_frame)
                detected_text = " ".join([res[1] for res in results]) if results else ""
            else:  # pytesseract
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_text = pytesseract.image_to_string(gray_frame)

            if not detected_text.strip():
                print(f"[{method}] No text detected.")
                await speak("No text detected")
                continue

            print(f"[{method}] {detected_text}")
            await speak(detected_text, True)

        except KeyboardInterrupt:
            print("Exiting...")
            break

# --- Run ---
asyncio.run(main())

