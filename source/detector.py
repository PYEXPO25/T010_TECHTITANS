import cv2
from multiple_faces import detect_multiple_faces
from eye_movement import detect_eye_movement
from head_movement import detect_head_movement

class Monitor:
    def __init__(self):
        self.suspicious_logs = []
        self.is_running = False

    def start_monitoring(self):
        """Start the monitoring process."""
        self.is_running = True

    def stop_monitoring(self):
        """Stop the monitoring process."""
        self.is_running = False

    def check_activity(self):
        """Check for suspicious activity (e.g., multiple faces, eye or head movement)."""
        if not self.is_running:
            return False, "Monitoring is not running."

        # Capture a frame from the webcam
        cap = cv2.VideoCapture(0)
        try:
            ret, frame = cap.read()
            if not ret:
                return False, "Failed to capture frame."
        finally:
            cap.release()

        # Detect multiple faces
        face_activity, face_message = detect_multiple_faces(frame)
        if face_activity:
            self.suspicious_logs.append(face_message)
            return True, face_message

        # Detect eye movement
        eye_activity, eye_message = detect_eye_movement(frame)
        if eye_activity:
            self.suspicious_logs.append(eye_message)
            return True, eye_message

        # Detect head movement
        head_activity, head_message = detect_head_movement(frame)
        if head_activity:
            self.suspicious_logs.append(head_message)
            return True, head_message

        return False, "No suspicious activity detected."

    def get_logs(self):
        """Retrieve the logs of suspicious activities."""
        return self.suspicious_logs