import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("🌲 Random Forest Training Started...")

DATASET_PATH = r"C:\Rishika\MajorProject_1\keypoints"

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

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("✅ Accuracy :", accuracy_score(y_test, y_pred))
print("✅ Precision:", precision_score(y_test, y_pred, average='weighted'))
print("✅ Recall   :", recall_score(y_test, y_pred, average='weighted'))
print("✅ F1 Score :", f1_score(y_test, y_pred, average='weighted'))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.savefig("RF_confusion_matrix.png")
plt.show()