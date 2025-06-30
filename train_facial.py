# train_facial.py

import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from src.facial.model_facial import build_facial_emotion_model
from src.facial.preprocess import load_dataset

print("[INFO] Loading data...")
dataset_path = "data/video/FER2013"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

X_train, X_test, y_train, y_test = load_dataset(dataset_path)
if len(X_train) == 0:
    raise ValueError("No training data found. Check dataset structure and image loading.")

print("[INFO] Building model...")
model = build_facial_emotion_model()
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("[INFO] Training model...")
checkpoint = ModelCheckpoint("models/facial_model.h5", save_best_only=True, monitor='val_accuracy')
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=20, batch_size=32, callbacks=[checkpoint])
