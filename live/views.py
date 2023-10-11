from django.shortcuts import render
from django.http import StreamingHttpResponse
from live.cameras import VideoCamera


# Create your views here.

def index(request):
    return render(request, 'index.html')

def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()), content_type="multipart/x-mixed-replace;boundary=frame")

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

