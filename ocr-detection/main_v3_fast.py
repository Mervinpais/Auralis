import cv2
import easyocr
import asyncio
import subprocess
from datetime import datetime

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

# --- TTS using espeak ---
async def speak(text):
    try:
        subprocess.run(["espeak", "-ven+m3", "-s140", text], check=True)
    except Exception as e:
        print(f"[ERR] TTS playback error: {e}")

# --- Main loop ---
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

            # Grayscale + histogram equalization for better contrast
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            # Convert back to RGB (EasyOCR expects 3 channels)
            processed_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            results = reader.readtext(processed_frame)

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

