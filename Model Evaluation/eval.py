import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# PATHS (FIXED)
# =========================
KEYPOINT_DIR = r"C:\Rishika\MajorProject_1\keypoints"
MODEL_PATH  = r"C:\Rishika\MajorProject_1\Model Training\isl_keypoint_model_2.h5"

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
            kp = np.load(os.path.join(cls_path, f))

            # safety check
            if kp.shape != (63,):
                continue

            X.append(kp)
            y.append(label_map[cls])

X = np.array(X)
y = np.array(y)

print("✅ Loaded data:")
print("X shape:", X.shape)
print("y shape:", y.shape)

if len(X) == 0:
    raise ValueError("❌ No data loaded. Check keypoint directory.")

# =========================
# SAME SPLIT LOGIC AS TRAINING
# =========================
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42,
    stratify=y
)

print("✅ Test data shape:", X_test.shape, y_test.shape)

# =========================
# LOAD TRAINED MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# PREDICTIONS
# =========================
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# =========================
# EVALUATION METRICS
# =========================
print("\n✅ Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred, target_names=classes))

print("\n✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
