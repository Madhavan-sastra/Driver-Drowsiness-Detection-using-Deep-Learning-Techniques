document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const alertDiv = document.getElementById('alert');

    // Get user media
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            video.srcObject = stream;
        })
        .catch(function(error) {
            console.error('Error accessing webcam:', error);
        });

    // Process video frames
    setInterval(processVideo, 1000); // Adjust the interval as needed

    function processVideo() {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL('image/jpeg');

        fetch('/process_video', {
            method: 'POST',
            body: JSON.stringify({ image_data: imageData }),
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            // Handle response from backend
            console.log('Response from backend:', data);
            if (data.score > 15) {
                alertDiv.style.display = 'block';
            } else {
                alertDiv.style.display = 'none';
            }
        })
        .catch(error => console.error('Error processing video frames:', error));
    }
});
