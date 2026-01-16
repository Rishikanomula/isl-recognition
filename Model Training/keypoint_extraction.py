import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

DATASET_DIR = "C:\Rishika\MajorProject_1\Indian"
OUTPUT_DIR = "C:\Rishika\MajorProject_1\keypoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)
# extracting keypoints
for class_name in sorted(os.listdir(DATASET_DIR)):
    class_path = os.path.join(DATASET_DIR, class_name)
    out_class_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(out_class_path, exist_ok=True)

    if not os.path.isdir(class_path):
        continue

    print(f"Processing class: {class_name}")

    for img_name in tqdm(os.listdir(class_path)):
        if not img_name.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(class_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # If no hand detected, skip
        if not results.multi_hand_landmarks:
            continue

        hand_landmarks = results.multi_hand_landmarks[0]

        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])

        keypoints = np.array(keypoints)

        # Save keypoints
        out_file = img_name.replace(".jpg", ".npy")
        np.save(os.path.join(out_class_path, out_file), keypoints)

hands.close()
