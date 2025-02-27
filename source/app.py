from flask import Flask, render_template, Response, jsonify, request
import cv2
from math import sqrt, atan2
import mediapipe as mp
import numpy as np
import threading
import time

app = Flask(__name__)

# MediaPipe setup
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# MediaPipe Face Detection and Mesh setup
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=3,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_head_pose(landmarks, frame):
    """Calculate head pose angle and distance based on facial landmarks."""
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[133]

    nose_x, nose_y = int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0])
    left_eye_x, left_eye_y = int(left_eye.x * frame.shape[1]), int(left_eye.y * frame.shape[0])
    right_eye_x, right_eye_y = int(right_eye.x * frame.shape[1]), int(right_eye.y * frame.shape[0])

    dx = right_eye_x - left_eye_x
    dy = right_eye_y - left_eye_y
    distance = sqrt(dx ** 2 + dy ** 2)

    angle = atan2(dy, dx) * 180.0 / np.pi
    return angle, distance

def detect_multiple_faces(results):
    """Detect number of faces in the frame."""
    if results.detections:
        return len(results.detections)
    return 0

class Monitor:
    def __init__(self):
        self.suspicious_logs = []
        self.is_running = False
        self.cap = None
        self.frame = None
        self.processed_frame = None
        self.lock = threading.Lock()
        
    def start_monitoring(self):
        """Start the monitoring process."""
        with self.lock:
            if not self.is_running:
                self.is_running = True
                self.cap = cv2.VideoCapture(0)
                threading.Thread(target=self.process_frames, daemon=True).start()
                return True
            return False

    def stop_monitoring(self):
        """Stop the monitoring process."""
        with self.lock:
            self.is_running = False
            if self.cap and self.cap.isOpened():
                self.cap.release()
            return True

    def process_frames(self):
        """Process frames in a continuous loop."""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Create a copy of the frame for processing
            with self.lock:
                self.frame = frame.copy()
                
            # Process the frame for suspicious activity
            suspicious, reason, processed_frame = self.analyze_frame(frame)
            
            # Store the processed frame
            with self.lock:
                self.processed_frame = processed_frame
                
            # Log suspicious activity
            if suspicious:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                self.suspicious_logs.append({"timestamp": timestamp, "reason": reason})
            
            time.sleep(0.03)  # Prevent high CPU usage

    def analyze_frame(self, frame):
        """Analyze a frame for suspicious activity."""
        # Copy frame to avoid modification issues
        display_frame = frame.copy()
        
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face detection
        face_results = face_detection.process(rgb_frame)
        
        # Process face mesh
        face_mesh_results = face_mesh.process(rgb_frame)
        
        suspicious = False
        reason = ""

        # If faces are detected, highlight them
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # If face mesh landmarks are detected
        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                # Get head pose angle and distance
                angle, distance = get_head_pose(face_landmarks.landmark, frame)
                
                # Check for abnormal head movements
                if abs(angle) > 15 or distance > 100:
                    cv2.putText(display_frame, "Suspicious Head Movement!", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    suspicious = True
                    reason = "Suspicious Head Movement Detected"
                
                # Eye movement detection
                right_eye_center = face_landmarks.landmark[362]
                left_eye_center = face_landmarks.landmark[33]
                
                right_eye_position = (int(right_eye_center.x * frame.shape[1]), 
                                     int(right_eye_center.y * frame.shape[0]))
                left_eye_position = (int(left_eye_center.x * frame.shape[1]), 
                                    int(left_eye_center.y * frame.shape[0]))
                
                cv2.circle(display_frame, right_eye_position, 5, (0, 255, 0), -1)  # Right eye
                cv2.circle(display_frame, left_eye_position, 5, (0, 0, 255), -1)   # Left eye

        # Check for multiple face detection
        num_faces = detect_multiple_faces(face_results)
        if num_faces > 1:
            cv2.putText(display_frame, f"Warning! {num_faces} Faces Detected", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            suspicious = True
            reason = f"Multiple Faces Detected: {num_faces}"

        # Add monitoring status
        cv2.putText(display_frame, "Monitoring Active", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return suspicious, reason, display_frame

    def check_activity(self):
        """Check for suspicious activity in the current frame."""
        if not self.is_running:
            return False, "Monitoring is not running."
            
        with self.lock:
            if self.frame is None:
                return False, "No frame available."
                
            # Get the most recent processed result
            if len(self.suspicious_logs) > 0:
                last_log = self.suspicious_logs[-1]
                return True, last_log["reason"]
            
        return False, "No suspicious activity detected."

    def get_logs(self):
        """Retrieve the logs of suspicious activities."""
        return self.suspicious_logs
        
    def get_frame(self):
        """Get the most recent processed frame."""
        with self.lock:
            if self.processed_frame is not None:
                return self.processed_frame
            elif self.frame is not None:
                return self.frame
        return None

# Initialize the monitor
monitor = Monitor()

def generate_frames():
    """Generate frames for the video feed."""
    while True:
        frame = monitor.get_frame()
        if frame is None:
            # Provide a blank frame when no camera feed is available
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not started", (150, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        # Yield the frame in the proper format for Flask Response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.03)  # Control frame rate

# Flask routes
@app.route("/")
def login():
    return render_template("login.html")

@app.route("/exam", methods=['POST'])
def exam():
    if request.method == "POST":
        return render_template("monitor.html")

@app.route("/video_feed")
def video_feed():
    """Route for streaming video feed."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/start_camera")
def start_camera():
    monitor.start_monitoring()
    return jsonify({"status": "Camera Started"})

@app.route("/stop_camera")
def stop_camera():
    monitor.stop_monitoring()
    return jsonify({"status": "Camera Stopped"})

@app.route("/check_suspicious")
def check_suspicious():
    suspicious, reason = monitor.check_activity()
    return jsonify({"suspicious": suspicious, "reason": reason})

@app.route("/report")
def report():
    return render_template("report.html", 
                          username="Student123", 
                          exam_name="Math Exam", 
                          suspicious_logs=monitor.get_logs())

if __name__ == "__main__":
    app.run(debug=True)