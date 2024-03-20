from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from imutils.video import VideoStream
import cv2
import imutils
import time
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils

app = Flask(__name__)
socketio = SocketIO(app)

camera_started = False
vs = None
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Update with the correct path

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_ear():
    global vs, detector, predictor, lStart, lEnd, rStart, rEnd

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # calculate the eye aspect ratio for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        return ear

    # return a default value if no face is detected
    return 0.0

def generate():
    global camera_started, vs

    if not camera_started:
        vs = VideoStream(src=0).start()
        camera_started = True

    while True:
        frame = vs.read()
        ret, jpeg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n")

        # Calculate EAR and send it through WebSocket
        ear = calculate_ear()
        socketio.emit("update_ear", ear)
        #time.sleep(0.5)  # Add a small delay to control the update frequency

@app.route("/")
def index():
    return render_template("ear.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    socketio.run(app, debug=True)
