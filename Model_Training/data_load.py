# load_data.py content (or pasted above split code)
import os
import numpy as np

KEYPOINT_DIR = "C:\Rishika\MajorProject_1\keypoints"

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
