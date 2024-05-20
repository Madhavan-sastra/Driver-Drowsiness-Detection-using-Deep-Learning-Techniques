
from flask import Flask, render_template, Response, request, stream_with_context
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import smtplib
import threading
import time
from twilio.rest import Client

app = Flask(__name__)

# Initialize mixer
mixer.init()
sound = mixer.Sound('alarm.wav')
message2 = ""

# Twilio configurations
account_sid = ''
auth_token = ''
client = Client(account_sid, auth_token)
twilio_phone_number = ''
recipient_phone_number = ''

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
video_capture = None  # Video capture will be initialized on button click

# Initialize a flag to keep track of whether an alert email has been sent or not
alert_sent = False

def send_email_alert():
    global message2

    # Email configurations
    sender_email = ""  # Enter your email
    receiver_email = ""  # Enter recipient email
    password = ""  # Enter your email password

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
    message2 = "Email alert sent" +"<br>"+message2

    update_message()
    # Close the connection
    server.quit()

def send_sms_alert():
    try:
        message = client.messages.create(
            body='The driver is detected as drowsy. Please check on them.',
            from_=twilio_phone_number,
            to=recipient_phone_number
        )
        print("Alert SMS sent!")
    except Exception as e:
        print("Error:", e)
        pass

def alert_thread():
    global alert_sent, score
    try:
        sound.play()
        send_email_alert()
        send_sms_alert()
        alert_sent = True
        while score > 5:  # Continuously check the score
            time.sleep(1)
        sound.stop()
        alert_sent = False  # Reset the flag after stopping the alert sound
        # Update the HTML element after the alert is sent
        #update_element()
        
    except Exception as e:
        print("Error: in alert_thread", e)



def detect_drowsiness(frame):
    global count, score, rpred, lpred, alert_sent
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
    thicc=0
    if score > 15 and not alert_sent:
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        threading.Thread(target=alert_thread).start()
        alert_sent = True
        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2

        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    elif score <= 5:  # Reset the flag if the score drops below 5
        alert_sent = False

    return frame

def gen_frames():
    while True:
        if video_capture is not None:
            success, frame = video_capture.read()
            if not success:
                break
            else:
                frame = detect_drowsiness(frame)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # global message1
    # #if request.method == 'POST':
    # message1 = message2
    return render_template('index.html', email_alert_sent=alert_sent, score=score)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/activate_face_capturing', methods=['POST'])
def activate_face_capturing():
    global video_capture
    video_capture = cv2.VideoCapture(0)  # Start capturing video
    return "Video capturing activated"

@app.route('/update_message', methods=['POST'])
def update_message():
    global message2
    
    # Update message2 as needed
    return message2


if __name__ == '__main__':
    app.run(debug=True)
