from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras.preprocessing.image import img_to_array
import cv2
from keras.models import load_model
import numpy as np
import os
from django.core.files.base import ContentFile
import uuid


def upload(request):
    return render(request, 'upload.html')

# Define the paths to your models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
detection_model_path = os.path.join(MODELS_DIR, 'haarcascade_frontalface_default.xml')
emotion_model_path = os.path.join(MODELS_DIR, 'model.hdf5')

# Load the models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgusted", "scared", "happy", "sad", "surprised", "neutral"]

def predict_emotion(image):
    # Load and process the image
    if image is None:
        return [("Invalid image file", {})]
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    results = []

    if len(faces) == 0:
        results.append(("No face detected", {}))
    else:
        for (fX, fY, fW, fH) in faces:
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotion_classifier.predict(roi)[0]
            label = EMOTIONS[preds.argmax()]

            probs = {emotion: float(prob) for emotion, prob in zip(EMOTIONS, preds)}
            results.append((label, probs))

    return results


def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        # Convert the uploaded image to an OpenCV format
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        fs = FileSystemStorage()

        # Generate a unique filename using uuid
        unique_filename = str(uuid.uuid4()) + '.' + uploaded_image.name.split('.')[-1]
        image_path = fs.save(unique_filename, uploaded_image)
        labeled_filename = None  # Initialize the variable

        if image is not None:
            faces = face_detection.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            if len(faces) == 0:
                results = [("No face detected", {})]
            else:
                # Create a copy of the image to draw on
                image_copy = image.copy()
                for (fX, fY, fW, fH) in faces:
                    face = image[fY:fY + fH, fX:fX + fW]
                    roi = cv2.resize(face, (64, 64))
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    preds = emotion_classifier.predict(roi)[0]
                    label = EMOTIONS[preds.argmax()]

                    cv2.putText(image_copy, label, (fX, fY - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (238, 164, 64), 1)
                    cv2.rectangle(image_copy, (fX, fY), (fX + fW, fY + fH), (238, 164, 64), 2)

                # Save the image with labels and contours
                image_with_labels = cv2.imencode(".jpg", image_copy)[1].tostring()
                image_file = ContentFile(image_with_labels)
                labeled_filename = 'labeled_' + unique_filename  # Add 'labeled_' prefix to the unique filename
                fs.save(labeled_filename, image_file)

                results = [("Faces detected", {})]
        else:
            results = [("Invalid image file", {})]

        context = {
            'original_image_path': fs.url(unique_filename),
            'labeled_image_path': fs.url(labeled_filename),
            'results': results
        }
        return render(request, 'emotion_classification.html', context)

    return render(request, 'emotion_classification.html')
