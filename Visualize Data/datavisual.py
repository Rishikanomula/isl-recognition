import os
import cv2
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

DATASET_DIR = "C:/Rishika/MajorProject_1/Indian"  # original image folder

classes = sorted(os.listdir(DATASET_DIR))[:10]  # show first 10 classes

plt.figure(figsize=(15, 6))

for i, cls in enumerate(classes):
    cls_path = os.path.join(DATASET_DIR, cls)
    img_name = os.listdir(cls_path)[0]  # first image
    img = cv2.imread(os.path.join(cls_path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.title(f"Class: {cls}")
    plt.axis("off")

plt.suptitle("Sample ISL Gesture Images")
plt.tight_layout()
plt.show()
