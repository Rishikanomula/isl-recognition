import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
from collections import deque

# =========================
# PATHS
# =========================
KEYPOINT_DIR = r"C:\Rishika\MajorProject_1\keypoints"
MODEL_PATH  = r"C:\Rishika\MajorProject_1\Model Training\isl_keypoint_model_2.h5"

# =========================
# LOAD LABELS
# =========================
labels = sorted([
    d for d in os.listdir(KEYPOINT_DIR)
    if os.path.isdir(os.path.join(KEYPOINT_DIR, d))
])

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# MEDIAPIPE HANDS (ONE HAND)
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# =========================
# KEYPOINT EXTRACTION (63)
# =========================
def extract_keypoints(results):
    keypoints = np.zeros(21 * 3)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        keypoints = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand.landmark]
        ).flatten()

    return keypoints

# =========================
# TEMPORAL SMOOTHING
# =========================
prediction_buffer = deque(maxlen=8)
CONFIDENCE_THRESHOLD = 0.75

# =========================
# WEBCAM LOOP
# =========================
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    keypoints = extract_keypoints(results)

    prediction = model.predict(
        np.expand_dims(keypoints, axis=0),
        verbose=0
    )[0]

    class_id = np.argmax(prediction)
    confidence = prediction[class_id]

    if confidence > CONFIDENCE_THRESHOLD:
        prediction_buffer.append(class_id)

        if prediction_buffer.count(class_id) > 5:
            cv2.putText(
                frame,
                f"{labels[class_id]} ({confidence:.2f})",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 255, 0),
                3
            )

    cv2.imshow("ISL Live Translation (Single-Hand Model)", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
