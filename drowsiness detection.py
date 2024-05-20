from flask import Flask, render_template, Response
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import smtplib
import threading
import time  # Import the time module

app = Flask(__name__)

# Initialize mixer
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascade files
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Labels for eye states
lbl = ['Close', 'Open']

# Load the pre-trained CNN model
model = load_model('models/cnncat2.h5')

# Get the current working directory
path = os.getcwd()

# Font for displaying text
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Initialize count and score variables
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]
video_capture = cv2.VideoCapture(0)

# Initialize a flag to keep track of whether an alert email has been sent or not
alert_sent = False

def send_email_alert():
    # Email configurations
    sender_email = "224003054@sastra.ac.in"  # Enter your email
    receiver_email = "maddydon79@gmail.com"  # Enter recipient email
    password = "Madhavan@123"  # Enter your email password

    # Connect to the SMTP server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, password)

    # Email content
    subject = "Drowsiness Alert!"
    body = "The driver is detected as drowsy. Please check on them."

    # Construct the email message
    message = f"Subject: {subject}\n\n{body}"

    # Send the email
    server.sendmail(sender_email, receiver_email, message)
    print("Alert email sent!")

    # Close the connection
    server.quit()

def alert_thread():
    global alert_sent  # Declare the flag as global
    try:
        sound.play()
        send_email_alert()  # Send email alert
        alert_sent = True  # Set the flag to True after sending the email
    except Exception as e:
        print("Error:", e)
        pass

def detect_drowsiness(frame):
    global count, score, rpred, lpred, alert_sent  # Declare the variables as global
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count += 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = np.argmax(model.predict(r_eye), axis=-1)
        if rpred[0] == 1:
            lbl = 'Open'
        if rpred[0] == 0:
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        if lpred[0] == 1:
            lbl = 'Open'
        if lpred[0] == 0:
            lbl = 'Closed'
        break

    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0

    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 15 and not alert_sent:  # Check if score is above threshold and alert has not been sent
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        threading.Thread(target=alert_thread).start()
        alert_sent = True  # Set the flag to True after triggering the alert
        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2

        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    elif score <= 15:  # Reset the flag if the score drops below threshold
        alert_sent = False

    return frame

def gen_frames():
    while True:
        success, frame = video_capture.read()  # read the camera frame
        if not success:
            break
        else:
            frame = detect_drowsiness(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
