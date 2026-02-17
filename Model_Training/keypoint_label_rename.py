import os
import numpy as np

# =========================
# PATH TO KEYPOINT FOLDERS
# =========================
KEYPOINT_DIR = "C:\Rishika\MajorProject_1\keypoints"

X = []  # features
y = []  # labels

# =========================
# CREATE CLASS MAPPING
# =========================
classes = sorted([
    d for d in os.listdir(KEYPOINT_DIR)
    if os.path.isdir(os.path.join(KEYPOINT_DIR, d))
])

label_map = {cls: idx for idx, cls in enumerate(classes)}

print("Label Mapping:")
for k, v in label_map.items():
    print(f"{k} -> {v}")

# =========================
# LOAD KEYPOINTS + LABELS
# =========================
for cls in classes:
    cls_path = os.path.join(KEYPOINT_DIR, cls)

    for file in os.listdir(cls_path):
        if file.endswith(".npy"):
            kp = np.load(os.path.join(cls_path, file))
            X.append(kp)
            y.append(label_map[cls])

X = np.array(X)
y = np.array(y)

print("\nDataset Shapes:")
print("X:", X.shape)  # (samples, 63)
print("y:", y.shape)  # (samples,)
