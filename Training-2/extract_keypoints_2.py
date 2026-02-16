import os
import cv2
import numpy as np
import mediapipe as mp

# =========================
# PATHS
# =========================
DATASET_DIR = "C:/Rishika/MajorProject_1/Indian"   # image folders
OUTPUT_DIR = "C:/Rishika/MajorProject_1/Training-2/keypoints_2D"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.6
)

# =========================
# LOOP THROUGH DATASET
# =========================
classes = sorted(os.listdir(DATASET_DIR))

for cls in classes:

    class_input_path = os.path.join(DATASET_DIR, cls)
    class_output_path = os.path.join(OUTPUT_DIR, cls)

    os.makedirs(class_output_path, exist_ok=True)

    print(f"Processing class: {cls}")

    for img_name in os.listdir(class_input_path):

        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(class_input_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            keypoints = []

            # 🔥 Extract ONLY x and y (2D)
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])

            keypoints = np.array(keypoints)

            # Optional: wrist normalization (recommended)
            keypoints = keypoints.reshape(21, 2)
            keypoints = keypoints - keypoints[0]   # subtract wrist
            keypoints = keypoints.flatten()

            # Save as .npy
            output_path = os.path.join(
                class_output_path,
                img_name.replace(".jpg", ".npy").replace(".png", ".npy")
            )

            np.save(output_path, keypoints)

        else:
            print(f"Hand not detected in {img_name}")

hands.close()

print("✅ 2D keypoint extraction complete!")