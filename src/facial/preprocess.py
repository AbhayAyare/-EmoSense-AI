# src/facial/preprocess.py

import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
label_map = {label: idx for idx, label in enumerate(emotion_labels)}

def preprocess_image_from_path(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    return normalized.reshape(48, 48, 1)

def preprocess_image_from_array(face_array):
    gray = cv2.cvtColor(face_array, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    return normalized.reshape(1, 48, 48, 1)  # ready for prediction

def load_dataset(data_dir):
    images = []
    labels = []

    for emotion in emotion_labels:
        folder = os.path.join(data_dir, emotion)
        if not os.path.isdir(folder):
            continue

        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            try:
                img = preprocess_image_from_path(img_path)
                images.append(img)
                labels.append(label_map[emotion])
            except:
                continue

    X = np.array(images)
    y = to_categorical(np.array(labels), num_classes=len(emotion_labels))
    return train_test_split(X, y, test_size=0.2, random_state=42)
