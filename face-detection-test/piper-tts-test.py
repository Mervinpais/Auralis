import subprocess

def speak(text):
    try:
        subprocess.run([
            "piper", 
            "--model", "en_US-amy-medium.onnx", 
#            "--output_file", "output.wav", 
            "--rate", "100", 
            "--voice", "auto5#45", 
            "--language", "en-US", 
            text
        ])
    except Exception as e:
        print(f"TTS error: {e}")

speak("Hello")
