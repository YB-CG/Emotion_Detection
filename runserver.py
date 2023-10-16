import subprocess

try:
    subprocess.run(
        "python manage.py runserver", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
except KeyboardInterrupt:
    print("Server stopped.")
