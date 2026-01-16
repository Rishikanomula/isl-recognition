import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# PATHS (CHANGE THESE)
# =========================
KEYPOINT_DIR = "C:\Rishika\MajorProject_1\keypoints"
MODEL_PATH = "C:\Rishika\MajorProject_1\Model Training\isl_keypoint_model.h5"

# =========================
# LOAD KEYPOINT DATA
# =========================
X, y = [], []

classes = sorted([
    d for d in os.listdir(KEYPOINT_DIR)
    if os.path.isdir(os.path.join(KEYPOINT_DIR, d))
])

label_map = {cls: idx for idx, cls in enumerate(classes)}

for cls in classes:
    cls_path = os.path.join(KEYPOINT_DIR, cls)
    for f in os.listdir(cls_path):
        if f.endswith(".npy"):
            X.append(np.load(os.path.join(cls_path, f)))
            y.append(label_map[cls])

X = np.array(X)
y = np.array(y)

print("Loaded data:", X.shape, y.shape)

# =========================
# SAME SPLIT AS TRAINING
# =========================
_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42,
    stratify=y
)

print("Test data:", X_test.shape, y_test.shape)

# =========================
# LOAD TRAINED MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# PREDICTIONS
# =========================
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# =========================
# EVALUATION METRICS
# =========================
print("\n✅ Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))

print("\n✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
