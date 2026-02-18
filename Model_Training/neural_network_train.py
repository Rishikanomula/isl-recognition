import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

print("Starting Neural Network Training Pipeline...\n")

# =========================
# LOAD DATA
# ========================= bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
DATASET_PATH = r"C:\Rishika\MajorProject_1\keypoints"
print("Loading dataset from:", DATASET_PATH)

labels = sorted(os.listdir(DATASET_PATH))
print("Classes detected:", labels)

label_map = {label: idx for idx, label in enumerate(labels)}

X, y = [], []

for label in labels:
    print("Loading class:", label)
    class_path = os.path.join(DATASET_PATH, label)

    for file in os.listdir(class_path):
        if file.endswith(".npy"):
            kp = np.load(os.path.join(class_path, file))
            if kp.shape == (63,):
                X.append(kp)
                y.append(label_map[label])

print("Data loading complete")

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("Dataset shape:", X.shape)
print("Labels shape :", y.shape)

# =========================
# NORMALIZATION
# =========================
print("\nNormalizing features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)
print("Normalization complete")

# =========================
# TRAIN / TEST SPLIT
# =========================
print("\nSplitting dataset into training and testing...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Split complete")
print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

# =========================
# BUILD MODEL
# =========================
print("\nBuilding Neural Network model...")

model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model summary:")
model.summary()

# =========================
# TRAIN MODEL
# =========================
print("\nTraining Neural Network model...")
start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

end_time = time.time()
print("Neural Network training completed")
print("Training time: {:.2f} seconds".format(end_time - start_time))

# =========================
# EVALUATION
# =========================
print("\nEvaluating model on test data...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n===== NEURAL NETWORK RESULTS =====")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

# =========================
# CONFUSION MATRIX
# =========================
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Neural Network Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("NeuralNetwork_confusion_matrix.png")
plt.close()

print("Confusion matrix saved as NeuralNetwork_confusion_matrix.png")

# =========================
# SAVE MODEL
# =========================
print("\nSaving trained model...")
model.save("neural_network_model.h5")
print("Model saved as neural_network_model.h5")

print("\nNeural Network Training Pipeline Finished Successfully.")