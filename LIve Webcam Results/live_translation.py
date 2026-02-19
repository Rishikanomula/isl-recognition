import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from collections import deque
pred_buffer = deque(maxlen=20)

# LOAD TRAINED MODEL
model = tf.keras.models.load_model("C:\Rishika\MajorProject_1\Model_Training\isl_keypoint_model.h5")

# LABELS (MUST MATCH TRAINING ORDER)
labels = [
    "1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
]

# MEDIAPIPE HANDS
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

sentence = ""
last_char = ""
last_time = time.time()

# smoothing buffer
pred_buffer = deque(maxlen=20)

print("Press C to clear | Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #FIX 1: mirror correction
    frame = cv2.flip(frame, 1)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    current_char = ""

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        keypoints = []
        for lm in hand.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])

        keypoints = np.array(keypoints).reshape(1, 63)

        preds = model.predict(keypoints, verbose=0)
        confidence = np.max(preds)
        idx = np.argmax(preds)

        # FIX 2: confidence threshold
        if confidence > 0.85:
            pred_buffer.append(idx)
            if pred_buffer.count(idx) > 12:
                current_char = labels[idx]

            #FIX 3: temporal smoothing
            idx = max(set(pred_buffer), key=pred_buffer.count)
            current_char = labels[idx]

            #FIX 4: slow, stable sentence building
            if (
                    current_char != last_char and
                    time.time() - last_time > 1.5
            ):
                sentence += current_char
                last_char = current_char
                last_time = time.time()

        cv2.putText(
            frame,
            f"Sign: {current_char} ({confidence:.2f})",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # display sentence
    cv2.putText(
        frame,
        f"Sentence: {sentence}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 0),
        2
    )

    cv2.imshow("ISL Live Translation", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ""
        last_char = ""
        pred_buffer.clear()

cap.release()
cv2.destroyAllWindows()
hands.close()
