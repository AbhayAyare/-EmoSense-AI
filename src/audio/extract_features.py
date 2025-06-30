import os
import librosa
import numpy as np

def extract_features_from_audio(audio, sr=22050, n_mfcc=40, max_pad_len=174):
    """
    Extract MFCC features directly from a raw audio array.

    Args:
        audio (np.ndarray): Raw audio signal.
        sr (int): Sampling rate.
        n_mfcc (int): Number of MFCCs.
        max_pad_len (int): Max length in time frames (e.g., 174).

    Returns:
        np.ndarray: 1D flattened MFCC feature vector.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc.flatten()

def process_dataset(data_dir, save_dir=None):
    """
    Process all .wav files in a dataset directory.

    Args:
        data_dir (str): Root folder with Actor_*/ folders containing .wav files.
        save_dir (str): Optional folder to save MFCC arrays as .npy

    Returns:
        List of (mfcc_array, filename)
    """
    features = []

    for actor_folder in os.listdir(data_dir):
        actor_path = os.path.join(data_dir, actor_folder)
        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                file_path = os.path.join(actor_path, file)
                mfcc = extract_mfcc(file_path)

                if mfcc is not None:
                    features.append((mfcc, file))

                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, file.replace(".wav", ".npy"))
                        np.save(save_path, mfcc)

    return features
