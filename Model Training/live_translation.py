import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("isl_keypoint_model.h5")

# =========================
# LABEL MAP (VERY IMPORTANT)
# SAME ORDER AS TRAINING
# =========================
labels = [
    "1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
]

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6
)

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)

sentence = ""
last_char = ""
last_time = time.time()

print("Press 'C' to clear | 'Q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        keypoints = []
        for lm in hand.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])

        keypoints = np.array(keypoints).reshape(1, 63)

        preds = model.predict(keypoints, verbose=0)
        idx = np.argmax(preds)
        char = labels[idx]

        # add char every 1 second (avoid spam)
        if char != last_char and time.time() - last_time > 1:
            sentence += char
            last_char = char
            last_time = time.time()

        cv2.putText(frame, f"Sign: {char}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # show sentence
    cv2.putText(frame, f"Sentence: {sentence}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    cv2.imshow("ISL Live Translation", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ""

cap.release()
cv2.destroyAllWindows()
hands.close()
