// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const uploadForm = document.getElementById('uploadForm');
const resultsSection = document.getElementById('resultsSection');
const uploadedImage = document.getElementById('uploadedImage');
const uploadedImageFile = null;

// Drag and Drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('drag-over');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('drag-over');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelect(files[0]);
    }
});

// File Input Change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Handle File Selection
function handleFileSelect(file) {
    // Validate file type
    const allowedTypes = ['image/png', 'image/jpeg', 'image/gif', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
        showError('Invalid file type. Please upload an image (PNG, JPG, GIF, BMP)');
        return;
    }
    
    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showError('File is too large. Maximum size: 16MB');
        return;
    }
    
    // Store file for later use
    window.uploadedImageFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        uploadedImage.src = e.target.result;
        resultsSection.style.display = 'block';
        document.getElementById('successResult').style.display = 'none';
        document.getElementById('errorMessage').style.display = 'none';
        document.getElementById('loadingSpinner').style.display = 'flex';
        
        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }, 100);
        
        // Upload for prediction
        uploadForPrediction(file);
    };
    reader.readAsDataURL(file);
}

// Upload File for Prediction
function uploadForPrediction(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        handlePredictionResponse(data);
    })
    .catch(error => {
        console.error('Error:', error);
        showError('An error occurred: ' + error.message);
    });
}

// Handle Prediction Response
function handlePredictionResponse(data) {
    document.getElementById('loadingSpinner').style.display = 'none';
    
    if (data.error) {
        showError(data.error);
    } else {
        showSuccess(data);
    }
}

// Show Error
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    document.getElementById('successResult').style.display = 'none';
    document.getElementById('loadingSpinner').style.display = 'none';
}

// Show Success
function showSuccess(data) {
    // Display predicted letter
    document.getElementById('predictedLetter').textContent = data.letter;
    
    // Display confidence
    const confidence = Math.round(data.confidence * 100);
    const confidenceFill = document.getElementById('confidenceFill');
    confidenceFill.style.width = confidence + '%';
    confidenceFill.textContent = confidence + '%';
    document.getElementById('confidenceText').textContent = confidence + '%';
    
    // Display top 5 predictions
    displayTopPredictions(data.all_predictions);
    
    // Show results
    document.getElementById('successResult').style.display = 'block';
    document.getElementById('errorMessage').style.display = 'none';
}

// Display Top 5 Predictions
function displayTopPredictions(predictions) {
    // Sort predictions by confidence
    const sorted = Object.entries(predictions)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5);
    
    const listDiv = document.getElementById('topPredictionsList');
    listDiv.innerHTML = '';
    
    sorted.forEach(([label, confidence], index) => {
        const percentage = Math.round(confidence * 100);
        
        const html = `
            <div class="prediction-item">
                <span class="prediction-label">${index + 1}. ${label}</span>
                <div class="prediction-bar">
                    <div class="prediction-bar-fill" style="width: ${percentage}%"></div>
                </div>
                <span class="prediction-value">${percentage}%</span>
            </div>
        `;
        
        listDiv.innerHTML += html;
    });
}

// Reset Upload
function resetUpload() {
    fileInput.value = '';
    resultsSection.style.display = 'none';
    uploadBox.classList.remove('drag-over');
    window.uploadedImageFile = null;
}

// Download Result
function downloadResult() {
    const letter = document.getElementById('predictedLetter').textContent;
    const confidence = document.getElementById('confidenceText').textContent;
    
    // Create canvas with image and result
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    const img = new Image();
    img.src = uploadedImage.src;
    
    img.onload = function() {
        canvas.width = img.width + 100;
        canvas.height = img.height + 150;
        
        // Draw background
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw image
        ctx.drawImage(img, 50, 50);
        
        // Draw result
        ctx.fillStyle = '#333';
        ctx.font = 'bold 24px Arial';
        ctx.fillText(`Predicted Letter: ${letter}`, 50, img.height + 80);
        ctx.font = '18px Arial';
        ctx.fillText(`Confidence: ${confidence}`, 50, img.height + 110);
        
        // Download
        const link = document.createElement('a');
        link.href = canvas.toDataURL('image/png');
        link.download = `isl_recognition_${Date.now()}.png`;
        link.click();
    };
}

// Initial UI state
resultsSection.style.display = 'none';
