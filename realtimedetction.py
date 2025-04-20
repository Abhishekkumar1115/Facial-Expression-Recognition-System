import cv2
from keras.models import model_from_json
import numpy as np
import pyttsx3
import threading
import time


json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")


haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)


labels = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'neutral', 5: 'sad', 6: 'surprise'
}
colors = {
    'angry': (0, 0, 255),
    'disgust': (0, 255, 0),
    'fear': (255, 0, 255),
    'happy': (0, 255, 255),
    'neutral': (255, 255, 255),
    'sad': (255, 0, 0),
    'surprise': (0, 165, 255)
}
emojis = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😱',
    'happy': '😄', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'
}


engine = pyttsx3.init()
engine.setProperty('rate', 150)
last_spoken = None

def greet_user():
    greeting = "Hello! I'm your emotion detector assistant. Let's see how you're feeling today!"
    engine.say(greeting)
    engine.runAndWait()

greet_user()


def speak_emotion(text):
    global last_spoken
    if text != last_spoken:
        last_spoken = text
        threading.Thread(target=lambda: engine.say(text) or engine.runAndWait()).start()


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


webcam = cv2.VideoCapture(0)


while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    emotion = "None"

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        img = extract_features(roi)
        pred = model.predict(img)
        pred_idx = pred.argmax()
        emotion = labels[pred_idx]
        confidence = int(pred[0][pred_idx] * 100)

     
        color = colors[emotion]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        emoji = emojis[emotion]
        text = f"{emotion} ({confidence}%) {emoji}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

        speak_emotion(emotion)

    timestamp = time.strftime('%H:%M:%S')
    status_text = f"Detected: {emotion.upper()}  |  Time: {timestamp}"
    cv2.rectangle(frame, (0, frame.shape[0]-30), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
    cv2.putText(frame, status_text, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

 
    cv2.imshow("Emotion Detector - Made by Abhishek", frame)

 
    if cv2.waitKey(1) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()
