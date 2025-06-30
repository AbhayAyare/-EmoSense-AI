import os
import sys
import cv2
import time
import threading
import numpy as np
import streamlit as st
import sounddevice as sd
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# Path setup
sys.path.append(os.path.abspath("."))

from src.facial.preprocess import preprocess_image_from_array
from src.audio.extract_features import extract_features_from_audio

# === Load models ===
@st.cache_resource
def load_all_models():
    facial_model = load_model("models/facial_model.h5")
    audio_model = load_model("models/audio_model.h5")
    yolo_face = YOLO("models/yolov8n-face.pt")
    return facial_model, audio_model, yolo_face

facial_model, audio_model, yolo_face = load_all_models()

# === Constants ===
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
SAMPLERATE = 22050
DURATION = 2  # seconds
audio_data = None
lock = threading.Lock()
stop_stream = False

# === Audio Recording Thread ===
def record_audio():
    global audio_data, stop_stream
    while not stop_stream:
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
        st.warning(f"[Audio Error] {e}")
        return None

def predict_audio_emotion(features):
    try:
        if features is None:
            return "Unknown", None
        features = np.expand_dims(features, axis=0)
        pred = audio_model.predict(features, verbose=0)[0]
        emotion = EMOTIONS[np.argmax(pred)]
        return emotion, pred
    except Exception as e:
        st.error(f"[ERROR] Audio prediction failed: {e}")
        return "Error", None

def predict_facial_emotion(frame):
    try:
        results = yolo_face.predict(source=frame, conf=0.5, verbose=False)
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                face_img = preprocess_image_from_array(face)
                pred = facial_model.predict(face_img, verbose=0)[0]
                emotion = EMOTIONS[np.argmax(pred)]
                return emotion, pred, (x1, y1, x2, y2)
        return "No Face", None, None
    except Exception as e:
        st.error(f"[ERROR] Facial prediction failed: {e}")
        return "Error", None, None

# === Streamlit App UI ===
st.set_page_config(page_title="EmoSense-AI", layout="wide")
st.title("🎭 Real-Time Emotion Detection (Facial + Audio)")

col1, col2 = st.columns(2)
video_placeholder = col1.empty()
info_box = col2.empty()

if st.button("▶ Start Detection", key="start_button"):
    stop_stream = False
    threading.Thread(target=record_audio, daemon=True).start()
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and not stop_stream:
        ret, frame = cap.read()
        if not ret:
            continue

        facial_emotion, facial_prob, face_box = predict_facial_emotion(frame)
        audio_feat = get_audio_features()
        audio_emotion, audio_prob = predict_audio_emotion(audio_feat)

        # Draw box
        if face_box:
            x1, y1, x2, y2 = face_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 255, 100), 2)
            cv2.putText(frame, f"Facial: {facial_emotion}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Display emotions
        combined_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(combined_frame, channels="RGB")

        with info_box.container():
            st.markdown("### 🎤 Audio Emotion")
            st.success(audio_emotion)
            if audio_prob is not None:
                st.bar_chart(dict(zip(EMOTIONS, audio_prob)))

            st.markdown("### 🧠 Facial Emotion")
            st.success(facial_emotion)
            if facial_prob is not None:
                st.bar_chart(dict(zip(EMOTIONS, facial_prob)))

        if st.button("⏹ Stop Detection", key="stop_button_unique"):
            stop_stream = True
            break

    cap.release()
    cv2.destroyAllWindows()
    st.success("Stopped Emotion Detection.")

