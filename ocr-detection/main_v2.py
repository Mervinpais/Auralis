import cv2
import easyocr
import asyncio
import subprocess
import numpy as np
import os
import time

# --- Async Text-to-Speech ---
async def speak(text):
    await asyncio.to_thread(subprocess.run, ["espeak", "-ven+m3", "-s140", text])

# --- Show frame in terminal using catimg ---
async def show_frame_catimg(frame, width=50):
    temp_file = "tmp_preview.jpg"
    h, w = frame.shape[:2]
    aspect_ratio = h / w
    new_height = max(int(width * aspect_ratio), 1)
    preview = cv2.resize(frame, (width, new_height))
    cv2.imwrite(temp_file, preview)
    await asyncio.to_thread(subprocess.run, ["catimg", temp_file])
    os.remove(temp_file)

# --- OCR detection ---
async def ocr_frame(frame, reader):
    # EasyOCR expects RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb_frame)
    detected_text = []
    for bbox, text, conf in results:
        if text.strip():
            detected_text.append(text.strip())
    return " | ".join(detected_text)

# --- Main async loop ---
async def main():
    # Init EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=False)

    # Init camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # High-res capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Headless EasyOCR started. Press Ctrl+C to stop.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                await asyncio.sleep(1)
                continue

            # Terminal preview
            await show_frame_catimg(frame, width=50)

            # OCR detection
            text = await ocr_frame(frame, reader)
            if text:
                print("Detected text:")
                print(text)
                await speak(text)
                print("-" * 30)

            # Small delay to reduce CPU usage
            await asyncio.sleep(0.5)

    except KeyboardInterrupt:
        print("Stopping OCR...")
    finally:
        cap.release()

# --- Run the async loop ---
if __name__ == "__main__":
    asyncio.run(main())

