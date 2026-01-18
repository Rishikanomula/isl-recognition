import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os

# =========================
# PATHS
# =========================
KEYPOINT_DIR = "C:\Rishika\MajorProject_1\keypoints"
SCRATCH_MODEL = "C:\Rishika\MajorProject_1\Model Training\isl_alphabets_scratch.h5"      # from-scratch model
TRANSFER_MODEL = "C:\Rishika\MajorProject_1\Transfer Learning\isl_alphabets_transfer.h5"    # transfer learning model

# =========================
# LOAD ALPHABET DATA
# =========================
X, y = [], []

alphabet_classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
label_map = {cls: idx for idx, cls in enumerate(alphabet_classes)}

for cls in alphabet_classes:
    for f in os.listdir(os.path.join(KEYPOINT_DIR, cls)):
        if f.endswith(".npy"):
            X.append(np.load(os.path.join(KEYPOINT_DIR, cls, f)))
            y.append(label_map[cls])

X = np.array(X)
y = np.array(y)

# SAME SPLIT FOR FAIR COMPARISON
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# =========================
# LOAD MODELS
# =========================
scratch_model = tf.keras.models.load_model(SCRATCH_MODEL)
transfer_model = tf.keras.models.load_model(TRANSFER_MODEL)

# =========================
# PREDICTIONS
# =========================
y_pred_scratch = scratch_model.predict(X_test).argmax(axis=1)
y_pred_transfer = transfer_model.predict(X_test).argmax(axis=1)

# =========================
# ACCURACY
# =========================
acc_scratch = accuracy_score(y_test, y_pred_scratch)
acc_transfer = accuracy_score(y_test, y_pred_transfer)

print("\n📊 MODEL COMPARISON")
print("-------------------")
print("Alphabet (From Scratch):", acc_scratch)
print("Alphabet (Transfer Learning):", acc_transfer)
