import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# =========================
# PATH
# =========================
DATASET_PATH = r"C:\Rishika\MajorProject_1\keypoints"

# =========================
# LOAD DATA
# =========================
labels = sorted(os.listdir(DATASET_PATH))
label_map = {label: idx for idx, label in enumerate(labels)}

X, y = [], []

for label in labels:
    for file in os.listdir(os.path.join(DATASET_PATH, label)):
        if file.endswith(".npy"):
            kp = np.load(os.path.join(DATASET_PATH, label, file))
            if kp.shape == (63,):
                X.append(kp)
                y.append(label_map[label])

X = np.array(X, dtype=np.float32)
y = np.array(y)

# =========================
# NORMALIZE KEYPOINTS (IMPORTANT)
# =========================
X = (X - X.mean()) / (X.std() + 1e-8)

y = to_categorical(y)

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# MODEL (REDUCED CAPACITY)
# =========================
model = Sequential([
    Dense(128, activation='relu',
          kernel_regularizer=l2(0.001),
          input_shape=(63,)),
    Dropout(0.5),

    Dense(64, activation='relu',
          kernel_regularizer=l2(0.001)),
    Dropout(0.5),

    Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# EARLY STOPPING
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
)

# =========================
# TRAIN
# =========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60,          # upper limit
    batch_size=32,      # BEST choice here
    callbacks=[early_stop],
    verbose=1
)

model.save("isl_keypoint_model_2.h5")
print("✅ Model trained with reduced overfitting")


#evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\n✅ Final Test Accuracy:")
print(f"{test_accuracy * 100:.2f}%")

train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)

print("\n📊 Accuracy Comparison:")
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
