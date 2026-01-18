import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

KEYPOINT_DIR = "C:\Rishika\MajorProject_1\keypoints"

X, y = [], []

alphabet_classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
label_map = {cls: idx for idx, cls in enumerate(alphabet_classes)}

for cls in alphabet_classes:
    cls_path = os.path.join(KEYPOINT_DIR, cls)
    for f in os.listdir(cls_path):
        if f.endswith(".npy"):
            X.append(np.load(os.path.join(cls_path, f)))
            y.append(label_map[cls])

X = np.array(X)
y = np.array(y)

print("Loaded alphabet data:", X.shape, y.shape)

# =========================
# SPLIT
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

NUM_CLASSES = 26

model = Sequential([
    Dense(128, activation="relu", input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop]
)

# =========================
# SAVE MODEL
model.save("isl_alphabets_scratch.h5")
print("✅ Alphabet scratch model saved as isl_alphabets_scratch.h5")
