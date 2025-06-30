# src/fusion/model_fusion.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_fusion_model(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(512, input_shape=(input_dim,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))  # Multiclass output
    return model