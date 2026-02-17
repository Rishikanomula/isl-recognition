# 1️⃣ LOAD DATA
# (this creates X and y)
# ----------------------
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


# 2️⃣ SPLIT DATA
print("Data is being split ")
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


# TRAIN MODEL
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

print("Model Training Started ")
NUM_CLASSES = len(set(y))

model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32
)

model.save("isl_keypoint_model.h5")
print("Model is saved")
