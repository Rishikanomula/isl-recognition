import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

print("Starting Multi-Model Training Pipeline...\n")

# =========================
# LOAD DATA
# =========================
DATASET_PATH = r"C:\Rishika\MajorProject_1\keypoints"
print("Loading dataset from:", DATASET_PATH)

labels = sorted(os.listdir(DATASET_PATH))
print("Classes found:", labels)

label_map = {label: idx for idx, label in enumerate(labels)}

X, y = [], []

for label in labels:
    print(f" Loading class: {label}")
    class_path = os.path.join(DATASET_PATH, label)

    for file in os.listdir(class_path):
        if file.endswith(".npy"):
            kp = np.load(os.path.join(class_path, file))
            if kp.shape == (63,):
                X.append(kp)
                y.append(label_map[label])

print(" Data loading complete")

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(" Dataset Shape:", X.shape)
print(" Labels Shape:", y.shape)

# =========================
# NORMALIZE
# =========================
print("\n Normalizing keypoints...")
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(" Normalization complete")

# =========================
# SPLIT
# =========================
print("\n✂ Splitting dataset into train & test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(" Split complete")
print("   ➤ Training samples:", X_train.shape[0])
print("   ➤ Testing samples :", X_test.shape[0])

results = []

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_model(name, y_true, y_pred):

    print(f"\n Evaluating {name}...")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"   Accuracy : {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}")
    print(f"   F1 Score : {f1:.4f}")

    results.append([name, acc, prec, rec, f1])

    # Confusion Matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.close()

    print(f"Confusion matrix saved as {name}_confusion_matrix.png")

# =========================
# 1 RANDOM FOREST
# =========================
print("\n Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)
print(" Random Forest training complete")
y_pred_rf = rf.predict(X_test)
evaluate_model("RandomForest", y_test, y_pred_rf)

# =========================
#  2 SVM
# =========================
print("\n Training SVM...")
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
print(" SVM training complete")
y_pred_svm = svm.predict(X_test)
evaluate_model("SVM", y_test, y_pred_svm)

# =========================
# 3️ KNN
# =========================
print("\n Training KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(" KNN training complete")
y_pred_knn = knn.predict(X_test)
evaluate_model("KNN", y_test, y_pred_knn)

# =========================
# 4️ GRADIENT BOOSTING
# =========================
print("\n  Training Gradient Boosting...")
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
print("  Gradient Boosting training complete")
y_pred_gb = gb.predict(X_test)
evaluate_model("GradientBoosting", y_test, y_pred_gb)

# =========================
# 5️ NEURAL NETWORK
# =========================
print("\n  Training Neural Network...")
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

model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1)
print(" Neural Network training complete")

y_pred_nn = np.argmax(model.predict(X_test), axis=1)
evaluate_model("NeuralNetwork", y_test, y_pred_nn)

# =========================
# SAVE RESULTS
# =========================
print("\n Saving model comparison results...")
results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1 Score"
])

results_df.to_csv("model_comparison_results.csv", index=False)

print("\n===== FINAL MODEL COMPARISON =====")
print(results_df)

print("\n All models trained and evaluated successfully!")