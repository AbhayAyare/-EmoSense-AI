import sys
import os
import time
import threading

import cv2
import numpy as np
import sounddevice as sd
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import tensorflow as tf

sys.path.append(os.path.abspath('.'))

from src.facial.preprocess import preprocess_image_from_array
from src.audio.extract_features import extract_features_from_audio

# === Load Models ===
print("[INFO] Loading models...")
facial_model = load_model("models/facial_model.h5")
audio_model = load_model("models/audio_model.h5")
fusion_model = load_model("models/fusion_model.h5")
yolo_face = YOLO("models/yolov8n-face.pt")

# === Settings ===
emotion_labels = ['angry', 'disgust', 'happy', 'neutral', 'sad', 'surprise']
SAMPLERATE = 22050
DURATION = 2  # seconds
audio_data = None
lock = threading.Lock()

# === Audio Recording Thread ===
def record_audio_thread():
    global audio_data
    while True:
        audio = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1, dtype='float32')
        sd.wait()
        with lock:
            audio_data = audio.flatten()

def get_audio_features():
    global audio_data
    with lock:
        audio_copy = audio_data.copy() if audio_data is not None else None

    if audio_copy is None:
        return None

    try:
        return extract_features_from_audio(audio_copy, SAMPLERATE)
    except:
        return None

def get_facial_features(frame):
    results = yolo_face.predict(source=frame, conf=0.5, verbose=False)
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            try:
                face_img = preprocess_image_from_array(face)  # (1, 48, 48, 1)
                embedding_model = tf.keras.Model(
                    inputs=facial_model.input,
                    outputs=facial_model.layers[-2].output
                )
                feature = embedding_model.predict(face_img, verbose=0).flatten()
                return feature, (x1, y1, x2, y2)
            except Exception as e:
                print(f"[ERROR] Face embedding failed: {e}")
                continue
    return None, None

def fuse_predict(facial_feat, audio_feat):
    if facial_feat is None or audio_feat is None:
        return "Unknown"

    audio_feat = audio_feat.flatten()
    audio_feat = audio_feat / np.max(audio_feat + 1e-8)
    facial_feat = facial_feat / np.max(facial_feat + 1e-8)

    fusion_input = np.concatenate([audio_feat, facial_feat]).reshape(1, -1)
    try:
        pred = fusion_model.predict(fusion_input, verbose=0)
        emotion = emotion_labels[np.argmax(pred)]
        print("Predictions:", pred, "->", emotion)
        return emotion
    except Exception as e:
        print(f"[ERROR] Fusion prediction failed: {e}")
        return "Error"


# === Start Camera and Audio Thread ===
cap = cv2.VideoCapture(0)
threading.Thread(target=record_audio_thread, daemon=True).start()
print("[INFO] Starting real-time emotion fusion... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    facial_feat, face_box = get_facial_features(frame)
    audio_feat = get_audio_features()
    emotion = fuse_predict(facial_feat, audio_feat)

    if face_box:
        x1, y1, x2, y2 = face_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)

    cv2.imshow("Real-Time Fused Emotion Detection", frame)
    print("Audio max:", np.max(audio_feat), "Facial max:", np.max(facial_feat))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
