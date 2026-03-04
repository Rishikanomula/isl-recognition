# Configuration file for ISL Web App

import os

# Flask Configuration
DEBUG = True
SECRET_KEY = 'isl-recognition-secret-key-change-in-production'

# Upload Settings
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Model & Data Paths
MODEL_PATH = r"C:\Rishika\MajorProject_1\Model_Training\isl_keypoint_model.h5"
KEYPOINT_DIR = r"C:\Rishika\MajorProject_1\keypoints"

# MediaPipe Settings
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
MAX_NUM_HANDS = 1

# Server Settings
HOST = '127.0.0.1'
PORT = 5000
