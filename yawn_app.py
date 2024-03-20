from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from imutils.video import VideoStream
import cv2
import numpy as np
import imutils
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils

app = Flask(__name__)
socketio = SocketIO(app)

camera_started = False
vs = None
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Update with correct path

def calculate_yawn():
    global vs, detector, predictor
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        lip_distance = cal_yawn(shape)
        return lip_distance
    return 0.0

def cal_yawn(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = dist.euclidean(top_mean, low_mean)
    return distance

def generate():
    global camera_started, vs
    if not camera_started:
        vs = VideoStream(src=0).start()
        camera_started = True
    while True:
        frame = vs.read()
        ret, jpeg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n")
        yawn_distance = calculate_yawn()
        socketio.emit("update_yawn", yawn_distance)

@app.route("/")
def index():
    return render_template("yawn.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    socketio.run(app, debug=True)
