import numpy as np
import tensorflow as tf
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# =========================
# PATHS
# =========================
KEYPOINT_DIR = "C:/Rishika/MajorProject_1/keypoints"
MODEL_PATH = "C:\Rishika\MajorProject_1\Model Training\isl_keypoint_model.h5"

# =========================
# LOAD DATA (NUMBERS + ALPHABETS)
# =========================
X, y = [], []

classes = sorted([
    d for d in os.listdir(KEYPOINT_DIR)
    if os.path.isdir(os.path.join(KEYPOINT_DIR, d))
])

label_map = {cls: idx for idx, cls in enumerate(classes)}
inv_label_map = {v: k for k, v in label_map.items()}

for cls in classes:
    for f in os.listdir(os.path.join(KEYPOINT_DIR, cls)):
        if f.endswith(".npy"):
            X.append(np.load(os.path.join(KEYPOINT_DIR, cls, f)))
            y.append(label_map[cls])

X = np.array(X)
y = np.array(y)

# SAME SPLIT AS TRAINING
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# =========================
# LOAD MODEL + PREDICT
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    xticklabels=[inv_label_map[i] for i in range(len(classes))],
    yticklabels=[inv_label_map[i] for i in range(len(classes))],
    cmap="Blues",
    fmt="d"
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – ISL Numbers + Alphabets Model")
plt.tight_layout()
plt.show()
