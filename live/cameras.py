from pyvirtualdisplay import Display
display = Display(visible=0, size=(800, 600))
display.start()

import cv2
import os
from live.models import FacialExpressionModel
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, 'haarcascade_frontalface_default.xml')
MODEL_JSON_FILE = os.path.join(MODELS_DIR, 'model.json')
MODEL_WEIGHTS_FILE = os.path.join(MODELS_DIR, 'model_weights.h5')

model = FacialExpressionModel(MODEL_JSON_FILE, MODEL_WEIGHTS_FILE)
face_detection = cv2.CascadeClassifier(DETECTION_MODEL_PATH)

font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, frame = self.video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]
            roi = cv2.resize(face, (48, 48))
            prediction = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(frame, prediction, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
