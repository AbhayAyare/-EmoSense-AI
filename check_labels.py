# import numpy as np
# from collections import Counter

# print("[INFO] Loading labels...")
# y = np.load("data/processed/y_labels.npy")

# print(f"Total labels: {len(y)}")

# class_counts = Counter(y)
# cleaned_counts = {int(k): int(v) for k, v in class_counts.items()}
# print("📊 Class distribution:", cleaned_counts)

# import numpy as np

# X_audio = np.load("data/processed/X_audio.npy")
# X_facial = np.load("data/processed/X_facial.npy")
# y = np.load("data/processed/y_labels.npy")

# print(f"Audio:  {X_audio.shape}")
# print(f"Facial: {X_facial.shape}")
# print(f"Labels: {y.shape}")

# if len(X_audio) == len(X_facial) == len(y):
#     print("✅ Data is aligned properly!")
# else:
#     print("❌ Mismatch in sample counts.")

from tensorflow.keras.models import load_model

model = load_model("models/audio_model.h5")
model.summary()
