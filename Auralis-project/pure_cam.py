#!/usr/bin/env python3
# terminal_camera.py — Display webcam feed in terminal

import cv2
import numpy as np
from PIL import Image
import sys
import time

# ==== CONFIG ====
CAMERA_INDEX = 0       # 0 = default webcam
TERMINAL_WIDTH = 60    # number of characters wide
SLEEP_PER_FRAME = 0.01 # adjust for FPS control

def show_in_terminal(frame, width=60):
    """Display an OpenCV frame in terminal using ANSI colors"""
    # Convert BGR -> RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to terminal width
    h, w, _ = frame.shape
    aspect_ratio = h / w
    new_w = int(width)
    new_h = max(1, int(aspect_ratio * new_w * 0.55))  # adjust for terminal chars
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Convert to ANSI
    img_np = np.array(frame)
    lines = []
    for row in img_np:
        line = ""
        for pixel in row:
            r, g, b = pixel
            line += f"\x1b[48;2;{r};{g};{b}m "  # colored block
        line += "\x1b[0m"
        lines.append(line)

    # Move cursor to top-left and print
    print("\x1b[H", end="")
    print("\n".join(lines), end="")
    sys.stdout.flush()

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {CAMERA_INDEX}")
        return

    print("Press Ctrl+C to exit.")
    # Clear terminal once
    print("\x1b[2J", end="")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            show_in_terminal(frame, TERMINAL_WIDTH)
            time.sleep(SLEEP_PER_FRAME)
    except KeyboardInterrupt:
        print("\n🛑 Exiting.")
    finally:
        cap.release()

if __name__ == "__main__":
    main()

