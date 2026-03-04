import numpy as np
import cv2
import mediapipe as mp

# =========================
# MEDIAPIPE HANDS SETUP
# =========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # SINGLE HAND (matches training)
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# =========================
# KEYPOINT EXTRACTION
# =========================
def extract_keypoints(frame):
    """
    Extracts 63 hand keypoints (21 landmarks × 3 coords)
    from a BGR image frame.

    Returns:
        np.ndarray of shape (63,)
    """
    # Convert to RGB for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Default: no hand detected
    keypoints = np.zeros(21 * 3, dtype=np.float32)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32
        ).flatten()

    return keypoints
