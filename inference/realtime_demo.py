import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load YOLOv8 face detection model
face_model = YOLO("models/yolov8n-face.pt")  # Make sure the path is correct

# Load trained facial emotion classification model
emotion_model = load_model("models/facial_model.h5")

# Emotion label mapping
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def predict_emotion(face_img):
    """
    Preprocess the face image and predict emotion using trained model.
    """
    try:
        face_resized = cv2.resize(face_img, (48, 48))
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        norm = gray / 255.0
        reshaped = norm.reshape(1, 48, 48, 1)
        prediction = emotion_model.predict(reshaped, verbose=0)
        return emotion_labels[np.argmax(prediction)]
    except:
        return None

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Face detection using YOLO
    results = face_model.predict(source=frame, conf=0.3, verbose=False)

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            emotion = predict_emotion(face)
            if emotion:
                # Draw bounding box and emotion label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Real-Time Emotion Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
