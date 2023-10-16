from django.shortcuts import render
from django.http import StreamingHttpResponse
from live.cameras import VideoCamera
from django.contrib.auth.decorators import login_required


# Create your views here.
@login_required(login_url='login')  # Replace 'login' with the actual URL name of your login page
def index(request):
    return render(request, 'index.html')

def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()), content_type="multipart/x-mixed-replace;boundary=frame")

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

