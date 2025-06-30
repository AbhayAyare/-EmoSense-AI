import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from src.audio.extract_features import extract_mfcc
from src.audio.model_audio import build_cnn_model

# === Configuration ===
AUDIO_DIR = "data/audio/audios"
NUM_CLASSES = 6  # ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful']
EPOCHS = 30
BATCH_SIZE = 32
MODEL_SAVE_PATH = "models/audio_model.h5"

# === Emotion Mapping from RAVDESS
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
target_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful']

# === Load Data Function ===
def load_data(audio_dir):
    features = []
    labels = []

    for actor in os.listdir(audio_dir):
        actor_path = os.path.join(audio_dir, actor)
        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):
            if not file.endswith(".wav"):
                continue

            parts = file.split("-")
            emotion_code = parts[2]
            emotion = emotion_map.get(emotion_code)

            if emotion in target_emotions:
                filepath = os.path.join(actor_path, file)
                mfcc = extract_mfcc(filepath)

                if mfcc is not None:
                    mfcc = np.transpose(mfcc)  # Shape: (174, 40)
                    features.append(mfcc)
                    labels.append(target_emotions.index(emotion))

    X = np.array(features)                   # (samples, 174, 40)
    y = to_categorical(labels, num_classes=NUM_CLASSES)

    return train_test_split(X, y, test_size=0.2, random_state=42)

# === Train the CNN Model ===
print("[INFO] Loading and preprocessing audio files...")
X_train, X_test, y_train, y_test = load_data(AUDIO_DIR)

print(f"[INFO] Training data shape: {X_train.shape}")
print("[INFO] Building CNN model...")
model = build_cnn_model(input_shape=(174, 40), num_classes=NUM_CLASSES)

checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

print("[INFO] Starting training...")
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          callbacks=[checkpoint])

print(f"[INFO] Training complete. Best model saved to: {MODEL_SAVE_PATH}")
