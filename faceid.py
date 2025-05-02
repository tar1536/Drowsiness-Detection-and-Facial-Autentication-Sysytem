import cv2
import numpy as np
import os
import time

# Paths for Caffe model files
PROTOTXT_PATH = r"C:\Users\admin\Desktop\projects\drowsiness detection\deploy.prototxt"
MODEL_PATH = r"C:\Users\admin\Desktop\projects\drowsiness detection\res10_300x300_ssd_iter_140000.caffemodel"

# Dataset path
DATASET_PATH = "face_dataset"
TRAINED_MODEL_PATH = f"{DATASET_PATH}/face_trained_model.yml"
AUTHORIZED_USER_ID = "1"

# Load the face detection model
face_detector = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

# Check if OpenCV face module is available
try:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    print("[ERROR] OpenCV 'face' module is missing. Install opencv-contrib-python.")
    exit()

def get_camera_index():
    """Finds a working camera index."""
    for i in range(3):  # Check first 3 indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None

def capture_face():
    """Capture face data from different angles only after detecting movement."""
    cam_index = get_camera_index()
    if cam_index is None:
        print("[ERROR] No working camera found!")
        return

    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    cap = cv2.VideoCapture(cam_index)
    angles = ["Center", "Left", "Right", "Up", "Down"]
    count = 0
    images_per_angle = 6
    movement_threshold = 20  # Minimum movement to consider

    for angle in angles:
        time.sleep(3)  # Give user time to turn head

        prev_position = None
        movement_detected = False

        while not movement_detected:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Camera not working!")
                break
            
            frame = cv2.flip(frame, 1)  # Flip frame horizontally
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_detector.setInput(blob)
            detections = face_detector.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.6:
                    x, y, x1, y1 = (detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]).astype("int")
                    if x < 0 or y < 0 or x1 > frame.shape[1] or y1 > frame.shape[0]:
                        continue

                    if prev_position is not None:
                        movement = abs(prev_position[0] - x) + abs(prev_position[1] - y)
                        if movement > movement_threshold:
                            movement_detected = True
                            break
                    prev_position = (x, y)

            cv2.putText(frame, f"Move head: {angle}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Capturing Face", frame)
            cv2.waitKey(1)

        for _ in range(images_per_angle):
            count += 1
            face = gray[y:y1, x:x1]
            if face.shape[0] > 0 and face.shape[1] > 0:
                face = cv2.resize(face, (200, 200))
                cv2.imwrite(f"{DATASET_PATH}/{AUTHORIZED_USER_ID}_{count}.jpg", face)

    cap.release()
    cv2.destroyAllWindows()
    train_face_recognizer()

def train_face_recognizer():
    """Train the LBPH face recognizer using captured images."""
    images, labels = [], []
    for file in os.listdir(DATASET_PATH):
        if file.endswith(".jpg"):
            img_path = os.path.join(DATASET_PATH, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            label = int(file.split("_")[0])
            images.append(img)
            labels.append(label)

    if len(images) == 0:
        print("[ERROR] No training images found!")
        return

    face_recognizer.train(images, np.array(labels))
    face_recognizer.write(TRAINED_MODEL_PATH)
    print("[INFO] Face model trained and saved.")

def face_lock_system():
    """Recognize the face and grant access."""
    cam_index = get_camera_index()
    if cam_index is None:
        print("[ERROR] No working camera found!")
        return

    face_recognizer.read(TRAINED_MODEL_PATH)
    cap = cv2.VideoCapture(cam_index)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera not working!")
            break
        
        frame = cv2.flip(frame, 1)  # Flip frame horizontally
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_detector.setInput(blob)
        detections = face_detector.forward()

        access_granted = False

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                x, y, x1, y1 = (detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]).astype("int")
                if x < 0 or y < 0 or x1 > frame.shape[1] or y1 > frame.shape[0]:
                    continue
                
                face = gray[y:y1, x:x1]
                if face.shape[0] > 0 and face.shape[1] > 0:
                    face = cv2.resize(face, (200, 200))
                    label, confidence = face_recognizer.predict(face)

                    if str(label) == AUTHORIZED_USER_ID and confidence < 60:
                        access_granted = True
                        cv2.putText(frame, "Access Granted", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Access Denied", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if not access_granted:
            cv2.putText(frame, "Access Denied", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Face Lock System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    choice = input("1: Capture Face\n2: Run Face Lock System\nChoose: ")
    capture_face() if choice == "1" else face_lock_system()
