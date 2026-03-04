# Live Webcam ISL Translation

This folder contains a Streamlit web application for **real-time recognition** of Indian Sign Language letters and numbers using your webcam.

## Features

- 🎥 Live webcam feed with on-screen predictions
- 🤖 Fast translation using a trained CNN model and MediaPipe hand keypoints
- 📊 Displays predicted class and confidence directly on video
- ✅ Easy deployment with Streamlit

## Setup

1. Navigate to this folder:
   ```bash
   cd c:\Rishika\MajorProject_1\isl_web_app_live
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open the provided local URL (usually http://localhost:8501) in a browser with camera access.

## Notes

- Make sure your model file path in `app.py` is correct:
  ```python
  MODEL_PATH = r"C:\Rishika\MajorProject_1\Model_Training\isl_keypoint_model.h5"
  ```
- The app uses the `keypoints.py` module for MediaPipe hand landmark extraction; you can reuse the same file from the original web app.

## Troubleshooting

- If the camera doesn't start, ensure permissions are granted and the browser supports WebRTC.
- Improve lighting and hand visibility for more accurate predictions.
