import numpy as np
y = np.load("data/processed/y_labels.npy")
print("Class distribution:", np.sum(y, axis=0))
