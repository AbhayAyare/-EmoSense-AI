import os
import librosa
import numpy as np
import pandas as pd
import glob

# === CONFIG ===
AUDIO_DIR = "data/audio/audios"
LABELS_CSV = "data/audio/labels.csv"
OUTPUT_DIR = "data/processed"
SAMPLE_RATE = 22050
DURATION = 2  # seconds
N_MFCC = 40
MAX_LEN = 174  # ~2 sec MFCC frames

# === Load label mappings ===
df = pd.read_csv(LABELS_CSV)
emotion_map = {e: i for i, e in enumerate(sorted(df["emotion"].unique()))}

print(f"[INFO] Found {len(df)} audio label entries.")

# === Index all available .wav files (including in subfolders) ===
available_files = glob.glob(os.path.join(AUDIO_DIR, "**/*.wav"), recursive=True)
available_files = {os.path.basename(f): f for f in available_files}

print(f"[INFO] Indexed {len(available_files)} .wav files in {AUDIO_DIR}")

X_audio = []
y_labels = []
skipped = 0

for idx, row in df.iterrows():
    filename, emotion = row["filename"], row["emotion"]
    path = available_files.get(filename, None)

    if not path:
        print(f"[SKIP] File not found: {filename}")
        skipped += 1
        continue

    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc = np.transpose(mfcc)  # (time, n_mfcc)

        # Pad or trim to fixed length
        if mfcc.shape[0] < MAX_LEN:
            mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:MAX_LEN, :]

        X_audio.append(mfcc.flatten())  # Flatten to 1D
        y_labels.append(emotion_map[emotion])

    except Exception as e:
        print(f"[ERROR] {filename}: {e}")
        skipped += 1

# Convert to numpy arrays
X_audio = np.array(X_audio)
y_labels = np.array(y_labels)

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, "X_audio.npy"), X_audio)
np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), y_labels)

print(f"\n✅ [DONE] Processed: {len(X_audio)} samples")
print(f"❌ [SKIPPED]: {skipped} files missing or failed")
print(f"🧠 X_audio shape: {X_audio.shape}")
print(f"🏷️ y_labels shape: {y_labels.shape}")
