# Auralis

**Auralis** is a standalone wearable headgear designed to enhance situational awareness for visually impaired users. By combining a head-mounted camera, a Raspberry Pi 4B, and real-time audio feedback, Auralis provides information about the surrounding environment, such as faces, text, and objects.

---

## Features

- **Face Detection** – Announces when a face is detected nearby.  
- **Text Recognition (OCR)** – Reads out visible text in the environment.  
- **Object Awareness** – Optionally detects and notifies about objects.  
- **Hands-Free Operation** – Worn like a headband for fully mobile use.  
- **Standalone System** – Runs entirely on Raspberry Pi 4B without requiring a PC.
- **Offline** - Can be used offline
- **Private** - Data is processed completely locally
---

## Hardware Requirements

- Raspberry Pi 4B (4GB+ recommended)  
- Compatible USB webcam or Raspberry Pi camera module  
- Headband or adjustable strap for mounting the camera  
- Microphone and speaker (for audio feedback)  
- Portable battery pack (5V, 3A recommended)  

---

## Software Requirements

- Raspberry Pi OS (64-bit recommended)  
- Python 3.10+  
- OpenCV  
- PyTesseract (for OCR)  
- TensorFlow or PyTorch (for face/object detection)  
- Text-to-Speech library (e.g., `pyttsx3` or `gTTS`)  

---

## Installation

1. Clone the repository:  
```bash
git clone https://github.com/yourusername/auralis.git
cd auralis
```
### Install Python dependencies:
```python
pip install -r requirements.txt
```

Configure your camera in config.py (adjust resolution, device ID, etc.)

## Usage
```bash
python main.py
```

The device will start capturing video from the camera.

Detected faces, objects, or text will be announced through the speaker.

Adjust settings in config.py for language, detection sensitivity, or audio volume.


## Download links to models and such

| Model file | Download link |
|------------|--------------|
| retinaface.h5 | [GitHub release – retinaface.h5](https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5) |
| deploy.prototxt | [OpenCV sample deploy.prototxt](https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt) |
| res10_300x300_ssd_iter_140000.caffemodel | [OpenCV sample weights file](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel) |
| nn4.v2.t7 | [Main site for Downloads of models](https://cmusatyalab.github.io/openface/models-and-accuracies/)
| en_US-amy-medium.onnx | [en_US-amy-medium.onnx](https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx) |
| en_US-amy-medium.onnx.json | [en_US-amy-medium.onnx.json](https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json) |


## Contributing

Contributions are welcome! You can help with:

Optimizing detection models for Raspberry Pi

Adding support for more types of objects

Improving audio feedback clarity and latency

Extending multi-language support

## License

GPL License. See LICENSE for details.

## Special Thanks to the following;

- YOLO library and models

## Contact

For questions, suggestions, or collaboration:
Email: mervinpais14@protonmail.com
GitHub: github.com/Mervinpais
