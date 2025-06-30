# src/fusion/fuse_utils.py

def combine_emotions(facial_pred, audio_pred):
    """
    Simple rule-based fusion:
    - If both models agree → return the shared label
    - If they disagree → prioritize facial prediction (default)
    - You can improve this by adding confidence scores later.
    """
    if facial_pred == audio_pred:
        return facial_pred
    return facial_pred  # You can replace with `audio_pred` or apply custom logic
