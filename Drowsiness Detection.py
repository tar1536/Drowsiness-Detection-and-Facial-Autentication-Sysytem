from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import threading
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import requests
import os
from pydub import AudioSegment
from pydub.playback import play

# ========== CONFIGURATION ==========
BOT_TOKEN = "7889555199:AAGCBtoh7EGIMYiz3xR-613ItbSfDWOsg0o"
CHAT_IDS = ["6770502184", "1642067434", "5395592750", "1083949298"]
latest_coords = {'lat': None, 'lon': None}

# ========== TELEGRAM UTILITIES ==========
def send_telegram_alert(message):
    for chat_id in CHAT_IDS:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        try:
            requests.post(url, data=payload)
        except Exception as e:
            print(f"Failed to send message to {chat_id}: {e}")

def send_location(lat, lon):
    for chat_id in CHAT_IDS:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendLocation"
        payload = {"chat_id": chat_id, "latitude": lat, "longitude": lon}
        try:
            requests.post(url, data=payload)
        except Exception as e:
            print(f"Failed to send location to {chat_id}: {e}")

def get_current_location():
    lat = latest_coords.get('lat')
    lon = latest_coords.get('lon')
    if lat is not None and lon is not None:
        return lat, lon
    else:
        print("[WARNING] Location not yet received from browser.")
        return None, None

# ========== SOUND ALARM ==========
def sound_alarm(path, reason):
    global alarm_status, alarm_status2
    global drowsy_alert_count, yawn_alert_count
    global telegram_drowsy_sent, telegram_yawn_sent

    while alarm_status or alarm_status2:
        sound = AudioSegment.from_wav(path)
        play(sound)

        if reason == "drowsiness":
            drowsy_alert_count += 1
            if drowsy_alert_count == 2 and not telegram_drowsy_sent:
                send_telegram_alert("⚠️ Drowsiness detected twice! Stay alert.")
                lat, lon = get_current_location()
                if lat and lon:
                    send_location(lat, lon)
                telegram_drowsy_sent = True

        elif reason == "yawn":
            yawn_alert_count += 1
            if yawn_alert_count == 2 and not telegram_yawn_sent:
                send_telegram_alert("😴 Yawning detected twice! Consider a break.")
                lat, lon = get_current_location()
                if lat and lon:
                    send_location(lat, lon)
                telegram_yawn_sent = True

        break

# ========== DROWSINESS/YAWN UTILS ==========
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    return (leftEAR + rightEAR) / 2.0, leftEye, rightEye

def lip_distance(shape):
    top_lip = shape[48:55].tolist() + shape[60:65].tolist()
    low_lip = shape[54:61].tolist() + shape[64:68].tolist()
    return abs(np.mean(top_lip, axis=0)[1] - np.mean(low_lip, axis=0)[1]), np.array(top_lip + low_lip)

# ========== INIT ==========
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
ap.add_argument("-a", "--alarm", type=str, default="alarm.wav", help="path to alarm WAV file")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
YAWN_THRESH = 20

alarm_status = False
alarm_status2 = False
COUNTER = 0

drowsy_alert_count = 0
yawn_alert_count = 0
telegram_drowsy_sent = False
telegram_yawn_sent = False

# ========== LOAD MODELS ==========
script_dir = os.path.dirname(os.path.abspath(__file__))
predictor_path = os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# ========== FLASK SERVER ==========
app = Flask(__name__)
CORS(app)

@app.route('/stream')
def stream():
    def generate():
        while True:
            if alarm_status or alarm_status2:
                yield f"data: send_location\n\n"
                time.sleep(10)
            time.sleep(1)
    return Response(generate(), content_type='text/event-stream')

@app.route('/send_location', methods=['POST'])
def receive_location():
    global latest_coords
    data = request.get_json()
    lat = data.get('latitude')
    lon = data.get('longitude')
    if lat is not None and lon is not None:
        latest_coords['lat'] = lat
        latest_coords['lon'] = lon
        print(f"[INFO] Received browser location: {lat}, {lon}")
        return jsonify({'status': 'Location received'}), 200
    else:
        return jsonify({'error': 'Invalid data'}), 400

def run_flask():
    app.run(host='0.0.0.0', port=5000)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

print("[INFO] Flask server running at http://localhost:5000")

# ========== MAIN DRIVER LOOP ==========
while True:
    frame = vs.read()
    frame = cv2.flip(imutils.resize(frame, width=450), 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)

    for rect in rects:
        x, y, w, h = (rect.left(), rect.top(), rect.width(), rect.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        ear, leftEye, rightEye = final_ear(shape)
        distance, lip_hull = lip_distance(shape)

        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(lip_hull)], -1, (0, 255, 255), 2)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not alarm_status:
                    alarm_status = True
                    if args["alarm"]:
                        Thread(target=sound_alarm, args=(args["alarm"], "drowsiness"), daemon=True).start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            alarm_status = False

        if distance > YAWN_THRESH:
            if not alarm_status2:
                alarm_status2 = True
                if args["alarm"]:
                    Thread(target=sound_alarm, args=(args["alarm"], "yawn"), daemon=True).start()
            cv2.putText(frame, "YAWN ALERT!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            alarm_status2 = False

        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Driver Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
