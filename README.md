# Drowsiness-Detection-and-Facial-Autentication-Sysytem
 This project implements a real-time driver monitoring system that detects drowsiness and
 yawning using facial landmarks, and enhances security through a face recognition lock
 system, In case of drowsiness or yawning detection, it sends a Telegram alert with GPS
 location.
 
 Features:
 1. Drowsiness & Yawning Detection- Detects closed eyes and yawning based on facial landmarks using dlib.- Sounds an alarm and sends Telegram alerts with location 
    if drowsiness or yawning is detected twice.
 2. Location Sharing- A browser-based HTML client (`get_location.html`) automatically sends geolocation data to the
 3. Flask server.- Flask sends GPS location to predefined Telegram chat IDs.
 4. Face Lock System- Uses OpenCV's LBPH recognizer to authenticate user identity.- Trains on multiple facial angles and detects movement before capturing.- Grants 
    access only to authorized individuals.
 Project Structure
 
 1. trial.py               # Main script for drowsiness/yawn detection with Telegram/location alerts
 2. faceid.py              # Face recognition system (capture & access)
 3. get_location.html      # Browser-based geolocation sender
 4. alarm.wav              # Alarm sound file (required)
 5. shape_predictor_68_face_landmarks.dat  # Facial landmarks model (required)
 6. deploy.prototxt        # Face detection prototxt (required for faceid.py)
 7. res10_300x300_ssd_iter_140000.caffemodel  # Face detection model
 8. face_dataset/          # Stores training face images
 9. face_trained_model.yml # Trained LBPH face model
 How to Run
 1. Install Requirements
 pip install flask flask-cors imutils dlib opencv-python opencv-contrib-python numpy pydub requests
 Ensure `ffmpeg` is installed and accessible for `pydub` to play audio.
 2. Run Drowsiness Detection
 python trial.py --alarm alarm.wav- Open `get_location.html` in a browser to allow location sharing.- Press `q` to quit the video feed.
 3. Run Face Recognition
 python faceid.py
 Select:- `1`: To capture and train face data.- `2`: To launch the face lock authentication system.
 Security & Alerts- Alerts are sent to multiple users via Telegram using the bot token and chat IDs.- The system only sends alerts if drowsiness or yawning is 
 detected **twice**, reducing false
 positives.
 Deployment Notes- Ensure Flask server is accessible from the browser running `get_location.html` (default is
 `localhost:5000`).- Replace the bot token and chat IDs in `trial.py` with your own for real use.
 Credits- Facial landmark detection using `dlib`.- Face recognition via OpenCV LBPH algorithm.
- Real-time video processing with `imutils` and OpenCV.- Telegram alerts via `requests`.
