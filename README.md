# 🤖 EmoSense-AI

**EmoSense-AI** is a hybrid real-time emotion detection system that combines facial expression analysis and voice tone recognition to predict human emotions. It uses deep learning, computer vision, and signal processing to power affective computing.

---

## 🔥 Features

- 🎥 Real-time **facial emotion detection** using YOLOv8 and CNN
- 🎤 Real-time **audio emotion recognition** using MFCC features and Conv1D models
- 🔀 Option for **fusion model** combining both modalities
- 🖼️ Streamlit-based interactive web interface
- 📦 Modular project structure (easy to debug and extend)
- 💾 Works with `.h5` models trained on FER2013 & audio emotion datasets

---

## 🧠 Model Architecture

### Facial Model
- Input: 48x48 grayscale face image
- CNN + Dense layers
- Trained on FER2013

### Audio Model
- Input: (174, 40) MFCC features from 2s audio clip
- Conv1D + Dropout + Dense
- Trained on preprocessed audio dataset

### Fusion Model 
- Input: Flattened features from both models
- Dense layers → Output emotion

---

## 🗂️ Project Structure

EmoSense-AI/
├── app/
│ ├── templates/
│ │ └── index.html # (optional if running Streamlit only)
│ └── streamlit_app.py # Streamlit real-time interface
├── data/
│ ├── audio/ # Raw audio files
│ ├── video/FER2013/ # FER2013 facial images by emotion
│ └── processed/ # Preprocessed .npy feature files
├── inference/
│ ├── realtime_dual.py # Run facial+audio side-by-side (no fusion)
│ └── realtime_fused.py # Run fused model
├── models/
│ ├── facial_model.h5
│ ├── audio_model.h5
│ └── fusion_model.h5
├── src/
│ ├── facial/
│ │ └── preprocess.py
│ ├── audio/
│ │ └── extract_features.py
│ └── fusion/
│ ├── model_fusion.py
│ └── fuse_utils.py
├── train_fusion.py # Train fusion model
├── check_labels.py # Dataset label sanity checker
├── README.md
└── requirements.txt

Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
---
Install dependencies
pip install -r requirements.txt
---
Run the app (Streamlit interface)
streamlit run app/streamlit_app.py
🧪 Run Emotion Detection (CLI Scripts)
➤ Run Realtime Dual Emotion Recognition (facial + audio separately)
python inference/realtime_dual.py
➤ Run Realtime Fused Emotion Recognition (if fusion model is trained)
python inference/realtime_fused.py
---
🧰 Datasets Used
FER2013 (Facial)
RAVDESS / Custom Audio Dataset

📈 Sample Output
Facial Prediction	Audio Prediction	Final Output
😐 Neutral	😠 Angry	🟨 Dual Display
😢 Sad	😢 Sad	✅ Match Detected

