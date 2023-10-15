# from django.shortcuts import render
# from django.http import JsonResponse
# from live.models import FacialExpressionModel
# from django.core.files.storage import FileSystemStorage
# import cv2
# import numpy as np
# import os
# import imutils
# from keras.preprocessing.image import img_to_array


# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODELS_DIR = os.path.join(BASE_DIR, 'models')
# DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, 'haarcascade_frontalface_default.xml')
# MODEL_JSON_FILE = os.path.join(MODELS_DIR, 'model.json')
# MODEL_WEIGHTS_FILE = os.path.join(MODELS_DIR, 'model_weights.h5')

# model = FacialExpressionModel(MODEL_JSON_FILE, MODEL_WEIGHTS_FILE)
# face_detection = cv2.CascadeClassifier(DETECTION_MODEL_PATH)

# def upload_image(request):
#     if request.method == 'POST' and request.FILES.get('image_file'):
#         image_file = request.FILES['image_file']
#         fs = FileSystemStorage()
#         image_path = fs.save(image_file.name, image_file)
#         image_path = fs.url(image_path)

#         # Process the uploaded image and get emotion prediction
#         emotion = process_uploaded_image(image_path)
        
#         return JsonResponse({'emotion': emotion})
    
#     return render(request, 'upload.html')

# def upload_video(request):
#     if request.method == 'POST' and request.FILES.get('video_file'):
#         video_file = request.FILES['video_file']
#         fs = FileSystemStorage()
#         video_path = fs.save(video_file.name, video_file)
#         video_path = fs.url(video_path)

#         # Process the uploaded video and get emotion prediction
#         emotion = process_uploaded_video(video_path)
        
#         return JsonResponse({'emotion': emotion})
    
#     return render(request, 'upload.html')

# def process_uploaded_image(image_path):
#     if not os.path.isfile(image_path):
#         return None, "Invalid file path"
    
#     frame = cv2.imread(image_path)
#     frame = imutils.resize(frame, width=300)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_detection.detectMultiScale(gray, scaleFactor=1.1,
#                                             minNeighbors=5, minSize=(30, 30),
#                                             flags=cv2.CASCADE_SCALE_IMAGE)

#     frameClone = frame.copy()
#     frameClone = cv2.cvtColor(frameClone, cv2.COLOR_BGR2RGB)
#     emotions = []

#     for (fX, fY, fW, fH) in faces:
#         roi = gray[fY:fY + fH, fX:fX + fW]
#         roi = cv2.resize(roi, (64, 64))
#         roi = roi.astype("float") / 255.0
#         roi = img_to_array(roi)
#         roi = np.expand_dims(roi, axis=0)

#         preds = model.predict_emotion(roi)  # Use the model object
#         label = FacialExpressionModel.EMOTIONS_LIST[preds.argmax()]
#         emotions.append(label)

#         cv2.putText(frameClone, label, (fX, fY - 10),
#                     cv2.FONT_HERSHEY_DUPLEX, 1, (238, 164, 64), 1)
#         cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
#                       (238, 164, 64), 2)

#     return frameClone, emotions

# def process_uploaded_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frames = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame, emotions = process_uploaded_image(frame)

#         if emotions is not None:
#             frames.append((frame, emotions))

#     return frames

import os
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
import cv2
import numpy as np
from keras.models import load_model
from django.views.decorators.csrf import csrf_protect


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, 'haarcascade_frontalface_default.xml')
MODEL_WEIGHTS_FILE = os.path.join(MODELS_DIR, 'model_weights.h5')

def index(request):
    return render(request, 'upload.html')

@csrf_protect
def after(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file1']
        fs = FileSystemStorage()
        filename = fs.save('static/file.jpg', uploaded_file)

        img1 = cv2.imread(f'static/{filename}')
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(DETECTION_MODEL_PATH)  # Updated model path
        faces = cascade.detectMultiScale(gray, 1.1, 3)

        for x, y, w, h in faces:
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = img1[y:y + h, x:x + w]

        cv2.imwrite('static/after.jpg', img1)

        try:
            cv2.imwrite('static/cropped.jpg', cropped)
        except:
            pass

        try:
            image = cv2.imread(f'static/cropped.jpg', 0)
        except:
            image = cv2.imread(f'static/{filename}', 0)

        image = cv2.resize(image, (48, 48))
        image = image / 255.0
        image = np.reshape(image, (1, 48, 48, 1))

        model = load_model(MODEL_WEIGHTS_FILE)  # Updated model weights file

        prediction = model.predict(image)

        label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']

        prediction = np.argmax(prediction)
        final_prediction = label_map[prediction]

        # Return the result image filename in a JSON response
        response_data = {'result_image': 'static/after.jpg', 'data': final_prediction}
        return JsonResponse(response_data)

    return HttpResponse("Method not allowed")
