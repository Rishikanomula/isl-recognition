const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const captureCanvas = document.getElementById('capture');
const captureCtx = captureCanvas.getContext('2d');

let stream = null;
let intervalId = null;

startBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        startCapturing();
    } catch (err) {
        console.error('Camera not accessible:', err);
        alert('Unable to access camera. Please check permissions.');
    }
});

stopBtn.addEventListener('click', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    stopCapturing();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    overlay.textContent = '--';
});

function startCapturing() {
    intervalId = setInterval(() => {
        captureFrame();
    }, 300); // capture every 300ms
}

function stopCapturing() {
    if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
    }
}

async function captureFrame() {
    if (!video.videoWidth || !video.videoHeight) {
        return;
    }
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    captureCtx.drawImage(video, 0, 0);
    const dataUrl = captureCanvas.toDataURL('image/jpeg');

    try {
        const res = await fetch('/predict_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl })
        });
        const data = await res.json();
        if (data.error) {
            overlay.textContent = data.error;
        } else {
            const conf = Math.round(data.confidence * 100);
            overlay.textContent = `${data.letter} (${conf}%)`;
        }
    } catch (err) {
        console.error('Prediction error:', err);
        overlay.textContent = 'Error';
    }
}
