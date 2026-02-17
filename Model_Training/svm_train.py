import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("Starting SVM Training Pipeline...\n")

# =========================
# LOAD DATA
# =========================
DATASET_PATH = r"C:\Rishika\MajorProject_1\keypoints"
print("Loading dataset from:", DATASET_PATH)

labels = sorted(os.listdir(DATASET_PATH))
print("Classes detected:", labels)

label_map = {label: idx for idx, label in enumerate(labels)}

X, y = [], []

for label in labels:
    print(f"   ➤ Loading class: {label}")
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
print("\n Normalizing features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)
print("Normalization complete")

# =========================
# TRAIN / TEST SPLIT
# =========================
print("\n✂ Splitting dataset into training and testing...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(" Split complete")
print("   ➤ Training samples:", X_train.shape[0])
print("   ➤ Testing samples :", X_test.shape[0])

# =========================
# TRAIN SVM
# =========================
print("\n Initializing SVM model...")
model = SVC(kernel='rbf', verbose=True)   # verbose=True shows internal solver info

print("Training SVM model...")
start_time = time.time()

model.fit(X_train, y_train)

end_time = time.time()
print(" SVM training completed")
print(f"⏱ Training time: {end_time - start_time:.2f} seconds")

# =========================
# EVALUATION
# =========================
print("\n Evaluating model on test data...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n===== 📊 SVM RESULTS =====")
print(f"✅ Accuracy : {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall   : {recall:.4f}")
print(f"✅ F1 Score : {f1:.4f}")

# =========================
# CONFUSION MATRIX
# =========================
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("SVM_confusion_matrix.png")
plt.close()

print("Confusion matrix saved as SVM_confusion_matrix.png")

print("\nSVM Training Pipeline Finished Successfully!")