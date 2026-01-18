import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
# =========================
# PATHS
# =========================
KEYPOINT_DIR = "C:\Rishika\MajorProject_1\keypoints"
BASE_MODEL_PATH = "C:\Rishika\MajorProject_1\Transfer Learning\isl_numbers_base.h5"

# =========================
# LOAD ALPHABET DATA ONLY
# =========================
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
X = X - X.mean(axis=1, keepdims=True)

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

# =========================
# LOAD BASE MODEL
# =========================
base_model = tf.keras.models.load_model(BASE_MODEL_PATH)

# remove last layer (number classifier)
base_model = Model(
    inputs=base_model.input,
    outputs=base_model.layers[-2].output
)

# freeze base layers (TRANSFER LEARNING)
# freeze early layers only
for layer in base_model.layers[:-1]:
    layer.trainable = False

# allow last base layer to learn
base_model.layers[-1].trainable = True

# =========================
# NEW ALPHABET HEAD
# =========================
NUM_ALPHA_CLASSES = 26

output = Dense(NUM_ALPHA_CLASSES, activation="softmax", name="alphabet_output")(base_model.output)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),  # lower LR
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN (FEW EPOCHS ONLY)
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,          # minimal fine-tuning
    batch_size=32,
    callbacks=[early_stop]
)

# =========================
# SAVE MODEL
# =========================
model.save("isl_alphabets_transfer.h5")
print("✅ Alphabet transfer model saved as isl_alphabets_transfer.h5")
