# ISL Letter Recognition Web Application

A Flask-based web application for recognizing Indian Sign Language (ISL) letters from uploaded images.

## Features

- 🖼️ Image upload with drag-and-drop support
- 🤖 Real-time letter recognition using trained CNN model
- 📊 Confidence scores and top-5 predictions
- 📱 Responsive design (desktop & mobile)
- 💾 Download recognition results
- ⚡ Fast predictions with MediaPipe hand keypoints

## Project Structure

```
isl_web_app/
├── web_app.py              # Flask backend application
├── keypoints.py            # MediaPipe hand keypoint extraction
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Main HTML page
└── static/
    ├── style.css          # CSS styling
    └── script.js          # Frontend JavaScript
```

## Installation

### 1. Navigate to the project directory
```bash
cd c:\Rishika\MajorProject_1\isl_web_app
```

### 2. Activate virtual environment (if you have one)
```bash
venv\Scripts\Activate.ps1
```

### 3. Install required packages
```bash
pip install -r requirements.txt
```

## Running the Application

### Start the Flask development server
```bash
python web_app.py
```

You should see output like:
```
Model loaded: True
Labels: ['0', '1', '2', ..., 'Z']
Starting Flask app on http://127.0.0.1:5000
```

### Open in browser
Navigate to: `http://127.0.0.1:5000`

## Usage

1. **Upload Image**
   - Click the upload box or drag-and-drop an image
   - Supported formats: PNG, JPG, JPEG, GIF, BMP
   - Maximum file size: 16MB

2. **Get Prediction**
   - The model automatically processes the image
   - Displays the recognized letter with confidence score
   - Shows top-5 predictions

3. **Download Result**
   - Click "Download Result" to save the recognition result as an image

## Model Information

- **Type**: CNN (Convolutional Neural Network)
- **Input**: Hand keypoint data (21 landmarks × 3 coordinates)
- **Output**: 35 classes (A-Z letters + 0-9 numbers)
- **Accuracy**: 94.75%
- **Framework**: TensorFlow/Keras
- **Feature Extraction**: MediaPipe Hands

## API Endpoints

### POST `/predict`
Upload an image and get prediction

**Request:**
```bash
curl -X POST -F "file=@image.jpg" http://127.0.0.1:5000/predict
```

**Response:**
```json
{
    "letter": "A",
    "confidence": 0.9876,
    "error": null,
    "image": "data:image/jpeg;base64,..."
}
```

### GET `/`
Main web page

### GET `/health`
Health check endpoint

## Troubleshooting

### Model Not Loading
- Ensure the model file path is correct in `web_app.py`
- Check that `isl_keypoint_model.h5` exists at the specified path

### No Hand Detected
- Make sure the hand is clearly visible in the image
- Good lighting and contrast help with detection
- Ensure the entire hand is in the frame

### File Upload Issues
- Check maximum file size (16MB)
- Use supported image formats only
- Ensure proper permissions for the `uploads/` folder

## File Paths Configuration

If your model or keypoints are in different locations, update these paths in `web_app.py`:

```python
MODEL_PATH = r"C:\Rishika\MajorProject_1\Model_Training\isl_keypoint_model.h5"
KEYPOINT_DIR = r"C:\Rishika\MajorProject_1\keypoints"
```

## Dependencies

- **Flask**: Web framework
- **TensorFlow**: Deep learning framework
- **MediaPipe**: Hand tracking and keypoint extraction
- **OpenCV**: Image processing
- **NumPy**: Numerical computing
- **PIL**: Image manipulation

## Future Enhancements

- Real-time webcam input support
- Sentence recognition (sequential letters)
- Multi-hand support
- Model accuracy improvements
- Deployment to cloud platforms

## Author

Created for Indian Sign Language (ISL) Recognition Project

## License

This project is for educational purposes.
