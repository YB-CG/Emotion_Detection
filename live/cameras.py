import cv2
import os
from live.models import FacialExpressionModel
import numpy as np

# Get the path to the models directory
models_dir = os.path.dirname(os.path.abspath(__file__)) + '/models/'

# Define the paths to the model and CascadeClassifier files
model_json_file = os.path.join(models_dir, 'model.json')
model_weights_file = os.path.join(models_dir, 'model_weights.h5')
face_cascade_file = os.path.join(models_dir, 'haarcascade_frontalface_default.xml')

facec = cv2.CascadeClassifier(face_cascade_file)
model = FacialExpressionModel(model_json_file, model_weights_file)
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
