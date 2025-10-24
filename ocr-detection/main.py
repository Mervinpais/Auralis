import cv2
import pytesseract
import subprocess
import time
import os

def show_frame_catimg(frame, width=50):
    """
    Displays an OpenCV frame in the terminal using catimg.

    Args:
        frame: numpy array image
        width: width in pixels for the preview
    """
    temp_file = "tmp_preview.jpg"
    # Resize for terminal preview
    h, w = frame.shape[:2]
    aspect_ratio = h / w
    new_height = int(width * aspect_ratio)
    preview = cv2.resize(frame, (width, new_height))
    cv2.imwrite(temp_file, preview)
    # Display in terminal
    subprocess.run(["catimg", temp_file])
    # Clean up
    os.remove(temp_file)

# --- OCR function ---
def ocr_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Threshold to improve OCR
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray)
    return text.strip()

# --- Text-to-Speech ---
def speak(text):
    try:
        subprocess.run(["espeak", "-ven+m3", "-s140", text])
    except Exception as e:
        print(f"Speech error: {e}")

# --- Main function ---
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # High-resolution capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Headless OCR started. Press Ctrl+C to stop.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(1)
                continue

            # OCR
            text = ocr_frame(frame)
            show_frame_catimg(frame, 100)
            if text:
                print("Detected text:")
                print(text)
                speak(text)
                print("-" * 30)

            # Small delay to reduce CPU usage
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Stopping OCR...")

    finally:
        cap.release()

if __name__ == "__main__":
    main()

