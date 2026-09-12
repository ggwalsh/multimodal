# Multimodal

Webcam face detection and microphone speech-to-text. Written for AAI202 as the practical follow-on to my comparative case study on supervised vs unsupervised methods.

The paper is on my site: [The kiosk still has to be fair](https://geoffwalsh.xyz/notes/kiosk)

Haar cascades are old. This is the assignment as submitted, not a production detector. Speech goes out to Google’s Web Speech API — it needs a network.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python face_detector.py      # s save, q quit
python speech_recognizer.py
```

PyAudio is the painful one on macOS and Linux. A webcam and a microphone are required.

Geoff Walsh · AAI202
