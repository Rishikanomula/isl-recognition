import flask
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
import mediapipe as mp
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

# =========================
# FLASK APP SETUP
# =========================
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# =========================
# PATHS
# =========================
MODEL_PATH = r"C:\Rishika\MajorProject_1\Model_Training\isl_keypoint_model.h5"
KEYPOINT_DIR = r"C:\Rishika\MajorProject_1\keypoints"

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# =========================
# LOAD MODEL & LABELS
# =========================
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

labels = sorted([
    d for d in os.listdir(KEYPOINT_DIR)
    if os.path.isdir(os.path.join(KEYPOINT_DIR, d))
])

# =========================
# HELPER FUNCTIONS
# =========================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_keypoints(image_array):
    """
    Extract 63 hand keypoints (21 landmarks × 3 coords) from image.
    
    Args:
        image_array: numpy array (BGR format)
    
    Returns:
        np.ndarray of shape (63,)
    """
    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # Default: no hand detected
    keypoints = np.zeros(21 * 3, dtype=np.float32)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32
        ).flatten()
    
    return keypoints

def predict_letter(image_path):
    """
    Predict the letter from an image.
    
    Args:
        image_path: path to the image file
    
    Returns:
        dict with prediction results
    """
    if model is None:
        return {'error': 'Model not loaded', 'letter': None, 'confidence': None}
    
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Could not read image', 'letter': None, 'confidence': None}
        
        # Extract keypoints
        keypoints = extract_keypoints(image)
        
        # Check if hand was detected
        if np.all(keypoints == 0):
            return {'error': 'No hand detected in image', 'letter': None, 'confidence': None}
        
        # Reshape for model input
        keypoints = keypoints.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(keypoints, verbose=0)
        confidence = np.max(prediction[0])
        predicted_idx = np.argmax(prediction[0])
        predicted_letter = labels[predicted_idx]
        
        return {
            'error': None,
            'letter': predicted_letter,
            'confidence': float(confidence),
            'all_predictions': {labels[i]: float(prediction[0][i]) for i in range(len(labels))}
        }
    
    except Exception as e:
        return {'error': str(e), 'letter': None, 'confidence': None}

# =========================
# ROUTES
# =========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and return prediction.
    """
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use: PNG, JPG, JPEG, GIF, BMP'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get prediction
        result = predict_letter(filepath)
        
        # Read image and convert to base64 for display
        try:
            with open(filepath, 'rb') as img:
                img_base64 = base64.b64encode(img.read()).decode('utf-8')
                result['image'] = f"data:image/jpeg;base64,{img_base64}"
        except:
            pass
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'model_loaded': model is not None}), 200

# =========================
# ERROR HANDLERS
# =========================
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File is too large. Maximum size: 16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    print(f"Model loaded: {model is not None}")
    print(f"Labels: {labels}")
    print(f"Starting Flask app on http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
