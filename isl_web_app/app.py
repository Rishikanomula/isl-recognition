import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from keypoints import extract_keypoints

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ISL Live Translator",
    layout="wide"
)

# =========================
# PATHS
# =========================
MODEL_PATH = "C:\Rishika\MajorProject_1\Model_Training\isl_keypoint_model.h5"
KEYPOINT_DIR = "C:\Rishika\MajorProject_1\keypoints"

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =========================
# LOAD LABELS
# =========================
labels = sorted([
    d for d in os.listdir(KEYPOINT_DIR)
    if os.path.isdir(os.path.join(KEYPOINT_DIR, d))
])

# =========================
# SIDEBAR (RESULTS & INFO)
# =========================
st.sidebar.title("📊 Model Summary")
st.sidebar.write("**Task:** ISL Letters & Numbers Translation")
st.sidebar.write("**Model:** CNN (Keypoint-based)")
st.sidebar.write("**Input Features:** 63 hand landmarks")
st.sidebar.write("**Dataset:** ISL A–Z, 1–9")
st.sidebar.write("**Test Accuracy:** 94.75%")
st.sidebar.write("**Frameworks:** TensorFlow, MediaPipe")
st.sidebar.write("**Deployment:** Streamlit Web App")

# =========================
# MAIN TITLE
# =========================
st.title("Indian Sign Language Live Translation")
st.write(
    "Show an Indian Sign Language **letter or number** in front of your webcam."
)

# =========================
# VIDEO PROCESSOR
# =========================
class ISLVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # mirror for user comfort
        img = cv2.flip(img, 1)

        # extract keypoints (63)
        keypoints = extract_keypoints(img)

        # predict
        preds = model.predict(
            np.expand_dims(keypoints, axis=0),
            verbose=0
        )[0]

        class_id = np.argmax(preds)
        confidence = preds[class_id]
        label = labels[class_id]

        # draw prediction
        cv2.putText(
            img,
            f"{label} ({confidence:.2f})",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 255, 0),
            3
        )

        return img

# =========================
# START WEBCAM
# =========================
webrtc_streamer(
    key="isl-live",
    video_processor_factory=ISLVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "**Note:** This system uses real-time hand keypoints instead of raw images "
    "to achieve fast and accurate ISL translation."
)
