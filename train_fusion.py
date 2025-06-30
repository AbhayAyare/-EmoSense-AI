import numpy as np
import os
from sklearn.model_selection import train_test_split
from src.fusion.model_fusion import build_fusion_model
from tensorflow.keras.utils import to_categorical


# === Load data ===
print("[INFO] Loading .npy files...")
X_audio = np.load("data/processed/X_audio.npy")
X_facial = np.load("data/processed/X_facial.npy")
y = np.load("data/processed/y_labels.npy")  # Already one-hot

print(f"[INFO] Shapes - Audio: {X_audio.shape}, Facial: {X_facial.shape}, Labels: {y.shape}")

# === Normalize features ===
X_audio = X_audio / np.max(X_audio)
X_facial = X_facial / np.max(X_facial)

# === Align dataset ===
min_samples = min(len(X_audio), len(X_facial), len(y))
X_audio = X_audio[:min_samples]
X_facial = X_facial[:min_samples]
y = y[:min_samples]

# === Combine audio + facial ===
X_fusion = np.concatenate([X_audio, X_facial], axis=1)
print(f"[INFO] Combined fusion shape: {X_fusion.shape}")

# === Labels are already one-hot encoded
# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X_fusion, y, test_size=0.2, stratify=y, random_state=42)

# === One-hot encode labels ===
NUM_CLASSES = len(np.unique(y))
y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_test_cat = to_categorical(y_test, NUM_CLASSES)


# === Build fusion model ===
print("[INFO] Building fusion model...")
model = build_fusion_model(input_dim=X_fusion.shape[1], num_classes=NUM_CLASSES)

# === Compile and train ===
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("[INFO] Training...")
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# === Save model ===
os.makedirs("models", exist_ok=True)
model.save("models/fusion_model.h5")
print("✅ Saved model to models/fusion_model.h5")
