"""
Face Detector for Multimodal AI System
AAI202 Applications of Artificial Intelligence
Author: Geoff Walsh (A00186663)

Captures frames from the webcam, detects human faces using OpenCV's
Haar Cascade classifier, draws bounding boxes, and allows the user
to save annotated images.

Dependencies:
    opencv-python
    numpy

Usage:
    python face_detector.py

Controls:
    s - save current frame with detections
    q - quit
"""

import cv2
import os
from datetime import datetime


def load_face_cascade():
    """
    Load the pre-trained Haar Cascade classifier for frontal face detection.

    Returns:
        cv2.CascadeClassifier: Loaded face detector.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise IOError("Unable to load the face cascade classifier XML file.")
    return face_cascade


def detect_faces(frame, face_cascade, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """
    Detect faces in a BGR frame.

    Args:
        frame: Input image in BGR colour space.
        face_cascade: Pre-loaded Haar cascade classifier.
        scale_factor: How much the image size is reduced at each scale.
        min_neighbors: How many neighbours each candidate rectangle should have.
        min_size: Minimum object size. Smaller objects are ignored.

    Returns:
        faces: List of (x, y, w, h) rectangles.
        gray: Grayscale version of the frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return faces, gray


def draw_detections(frame, faces):
    """
    Draw green bounding boxes and labels on detected faces.
    """
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Face {i + 1}"
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return frame


def save_detection_image(frame, output_dir="face_detection_results"):
    """
    Save the annotated frame with a timestamped filename.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detected_face_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, frame)
    print(f"[INFO] Saved: {filepath}")
    return filepath


def main():
    """
    Open the webcam and run continuous face detection.
    """
    print("=" * 60)
    print("Face Detector - Webcam Capture")
    print("Press 's' to save current frame | Press 'q' to quit")
    print("=" * 60)

    try:
        face_cascade = load_face_cascade()
        print("[INFO] Haar Cascade loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load cascade: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        faces, _ = detect_faces(frame, face_cascade)
        display_frame = draw_detections(frame.copy(), faces)

        status = f"Faces: {len(faces)} | Saved: {saved_count}"
        cv2.putText(
            display_frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        cv2.imshow("Face Detection", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            save_detection_image(display_frame)
            saved_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Session ended. Total images saved: {saved_count}")


if __name__ == "__main__":
    main()
