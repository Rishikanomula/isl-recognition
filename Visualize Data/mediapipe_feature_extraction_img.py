import cv2
import mediapipe as mp
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

img_path = "C:/Rishika/MajorProject_1/Indian/A/0.jpg"  # any image

image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

hands = mp_hands.Hands(static_image_mode=True)
results = hands.process(image_rgb)

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(
            image_rgb,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

plt.figure(figsize=(5, 5))
plt.imshow(image_rgb)
plt.title("Hand Keypoints Extracted using MediaPipe")
plt.axis("off")
plt.show()
