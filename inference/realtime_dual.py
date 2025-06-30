import sys
import os
import threading
import cv2
import numpy as np
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO

sys.path.append(os.path.abspath('.'))

from src.facial.preprocess import preprocess_image_from_array
from src.audio.extract_features import extract_features_from_audio

# === Load Models ===
print("[INFO] Loading models...")
facial_model = load_model("models/facial_model.h5")
audio_model = load_model("models/audio_model.h5")
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
    except Exception as e:
        print(f"[ERROR] Audio feature extraction failed: {e}")
        return None

def extract_features_from_audio(audio, sr=22050, n_mfcc=40):
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc = mfcc.T  # Transpose to shape (time, features)

        # Ensure fixed length (pad or trim to 174)
        if mfcc.shape[0] < 174:
            pad_width = 174 - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:174, :]
        
        return mfcc  # Final shape: (174, 40)
    except Exception as e:
        print(f"[ERROR] extract_features_from_audio: {e}")
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
                face_img = preprocess_image_from_array(face)
                pred = facial_model.predict(face_img, verbose=0)[0]
                emotion = emotion_labels[np.argmax(pred)]
                print(f"🧠 Facial Prediction: {emotion} (probs: {pred})")
                return emotion, (x1, y1, x2, y2)
            except Exception as e:
                print(f"[ERROR] Facial prediction failed: {e}")
                continue
    return "Unknown", None

def predict_audio_emotion(audio_feat):
    if audio_feat is None:
        print("[DEBUG] Audio features are None.")
        return "Unknown"
    try:
        print(f"[DEBUG] Extracted audio shape: {audio_feat.shape}")
        
        if audio_feat.shape != (174, 40):
            print(f"[WARNING] Audio shape mismatch: expected (174, 40), got {audio_feat.shape}")
            return "Unknown"

        audio_feat = audio_feat.reshape(1, 174, 40)  # Add batch dimension
        pred = audio_model.predict(audio_feat, verbose=0)[0]
        emotion = emotion_labels[np.argmax(pred)]
        print(f"🔊 Audio Prediction: {emotion} (probs: {pred})")
        return emotion
    except Exception as e:
        print(f"[ERROR] Audio prediction failed: {e}")
        return "Error"


# === Start Camera and Audio Thread ===
cap = cv2.VideoCapture(0)
threading.Thread(target=record_audio_thread, daemon=True).start()
print("[INFO] Starting dual emotion recognition... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    facial_emotion, face_box = get_facial_features(frame)
    audio_feat = get_audio_features()
    audio_emotion = predict_audio_emotion(audio_feat)

    if face_box:
        x1, y1, x2, y2 = face_box
        display_text = f"Facial: {facial_emotion} | Audio: {audio_emotion}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    cv2.imshow("Real-Time Dual Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
