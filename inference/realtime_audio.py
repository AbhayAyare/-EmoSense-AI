import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import time

# === Load Trained Model ===
print("[INFO] Loading audio emotion model...")
model = tf.keras.models.load_model("models/audio_model.h5")

# === Parameters ===
DURATION = 2  # seconds
SAMPLE_RATE = 22050
THRESHOLD = 0.5  # confidence threshold
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful']

# === Emotion Prediction Function ===
def predict_emotion(audio):
    try:
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
        mfcc = np.transpose(mfcc)  # Shape: (time, features)

        # Ensure consistent shape: (174, 40)
        if mfcc.shape[0] < 174:
            pad_width = 174 - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        elif mfcc.shape[0] > 174:
            mfcc = mfcc[:174, :]

        mfcc = np.expand_dims(mfcc, axis=0)  # Shape: (1, 174, 40)

        # Predict
        prediction = model.predict(mfcc, verbose=0)
        max_index = np.argmax(prediction)
        confidence = prediction[0][max_index]

        if confidence < THRESHOLD:
            return "Uncertain"

        return emotion_labels[max_index]

    except Exception as e:
        print(f"[ERROR] Failed to predict: {e}")
        return "Error"

# === Start Real-Time Mic Detection ===
print("[INFO] 🎙️ Real-time voice emotion detection started.")
print("[INFO] Speak into the mic. Press Ctrl+C to stop.")

try:
    while True:
        print("\n[Listening...]")
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()

        audio = audio.flatten()
        emotion = predict_emotion(audio)

        print(f"🧠 Detected Emotion: {emotion}")
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n[INFO] 🔴 Detection stopped by user.")

except Exception as e:
    print(f"[FATAL ERROR] {e}")
