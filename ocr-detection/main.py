# ocr_camera.py

import cv2
import pytesseract

# If tesseract is not in your PATH, specify the full path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_frame(frame):
    # Convert frame to grayscale for better OCR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def main():
    cap = cv2.VideoCapture(0)  # 0 = default camera
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        text = ocr_frame(frame)
        if text.strip():  # Only print if text is detected
            print("Detected text:")
            print(text)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

