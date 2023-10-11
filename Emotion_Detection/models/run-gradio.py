import gradio as gr
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import os

# parameters for loading data and images
detection_model_path = '/home/skay/Desktop/Emotion_detection/sahadatu/haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = '/home/skay/Desktop/Emotion_detection/sahadatu/models/model.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgusted", "scared", "happy", "sad", "surprised",
            "neutral"]

def predict(webcam_frame, file_frame):
    if webcam_frame is not None:
        frame = webcam_frame
    elif file_frame is not None:
        # Load the image data from the file path
        if os.path.isfile(file_frame):
            frame = cv2.imread(file_frame)
        else:
            return None, "Invalid file path"
    else:
        return None, "No input provided"

    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1,
                                            minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

    frameClone = frame.copy()
    frameClone = cv2.cvtColor(frameClone, cv2.COLOR_BGR2RGB)
    for (fX, fY, fW, fH) in faces:
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
        # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]

        probs = {}
        cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (238, 164, 64), 1)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                      (238, 164, 64), 2)

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            probs[emotion] = float(prob)

    if len(faces) == 1:
        return frameClone, probs
    elif len(faces) > 1:
        return frameClone, None
    else:
        return frameClone, "Can't find your face"

inp = [
    gr.inputs.Image(source="webcam", label="Your face"),
    gr.inputs.Image(type='filepath', label='Input Image')
]

out = [
    gr.outputs.Image(type="pil", label="Predicted Emotion"),
    gr.outputs.Label(num_top_classes=3, label="Top 3 Probabilities")
]

title = "Emotion Classification"
description = "How well can this model predict your emotions? Take a picture with your webcam, and it will guess if you are: happy, sad, angry, disgusted, scared, surprised, or neutral."

gr.Interface(predict, inp, out, live=True, capture_session=True, title=title,
             description=description).launch(share=True)
