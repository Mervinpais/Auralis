import torch
import cv2
import time
from datetime import datetime
import os
import subprocess
import easyocr
import hashlib
import tempfile
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

def speak(text, fast=False):
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



# Model loading
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

save_dir = 'runs/detect/exp'
os.makedirs(save_dir, exist_ok=True)

while True:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    time.sleep(0.5)
    if not ret:
        continue

    # Run inference
    results = model(frame)

    # Render detections on the image
    results.render()  # modifies results.ims in-place

    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_path = os.path.join(save_dir, f"{timestamp}.jpg")

    # Save the frame with detections
    for img in results.ims:  # now use .ims after render()
        cv2.imwrite(save_path, img)

    print(f"Saved: {save_path}")
    results.print()
    
    df = results.pandas().xyxy[0]  # dataframe of detections for this frame
    detected_objects = df['name'].tolist()
    
    if detected_objects:
        print("Detected:", detected_objects)
        speak(str(detected_objects), True)

    com = input("> ")
    if com == "show":
        subprocess.run(["catimg", save_path])
        input()
    else:
        continue


