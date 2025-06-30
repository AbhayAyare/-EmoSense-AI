import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model, Model
from tqdm import tqdm

# === CONFIG ===
ROOT_DIR = "data/video/FER2013"
MODEL_PATH = "models/facial_model.h5"
OUTPUT_DIR = "data/processed"
TARGET_SIZE = (48, 48)

# === Emotion labels (consistent order)
emotion_map = {'angry': 0, 'calm': 1, 'disgust': 2, 'fearful': 3,
               'happy': 4, 'neutral': 5, 'sad': 6, 'surprise': 7}

# === Load model & get embedding model (excluding softmax)
base_model = load_model(MODEL_PATH)
embedding_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

X_facial = []
y_labels = []

print("[INFO] Extracting facial features...")

for emotion_label in os.listdir(ROOT_DIR):
    folder_path = os.path.join(ROOT_DIR, emotion_label)
    if not os.path.isdir(folder_path) or emotion_label not in emotion_map:
        continue

    label_index = emotion_map[emotion_label]

    for img_name in tqdm(os.listdir(folder_path), desc=f"[{emotion_label}]"):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        try:
            path = os.path.join(folder_path, img_name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, TARGET_SIZE)
            img = img / 255.0
            img = img.reshape(1, 48, 48, 1)

            embedding = embedding_model.predict(img, verbose=0)
            X_facial.append(embedding.flatten())
            y_labels.append(label_index)
        except Exception as e:
            print(f"[ERROR] {img_name}: {e}")

X_facial = np.array(X_facial)
y_labels = np.array(y_labels)

# === Save to .npy
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, "X_facial.npy"), X_facial)
np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), y_labels)

print(f"\n✅ [DONE] Saved {X_facial.shape[0]} facial embeddings to X_facial.npy")
print(f"🧠 X_facial shape: {X_facial.shape}")
