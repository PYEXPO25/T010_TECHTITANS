import cv2

class Monitor:
    def __init__(self):
        self.suspicious_logs = []
        self.is_running = False

        # Load the pre-trained deep learning face detection model
        # self.net = cv2.dnn.readNetFromCaffe(
        #     "deploy.prototxt",  # Path to the prototxt file
        #     "res10_300x300_ssd_iter_140000.caffemodel"  # Path to the caffemodel file
        # )

    def start_monitoring(self):
        """Start the monitoring process."""
        self.is_running = True

    def stop_monitoring(self):
        """Stop the monitoring process."""
        self.is_running = False

    def check_activity(self):
        """Check for suspicious activity (e.g., multiple faces)."""
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

        # Get the frame dimensions
        (h, w) = frame.shape[:2]

        # Prepare the frame for face detection
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),  # Resize to 300x300 and normalize
            (104.0, 177.0, 123.0)  # Mean subtraction values for normalization
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        # Count the number of faces detected
        face_count = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > 0.5:  # Confidence threshold
                face_count += 1

        if face_count > 1:
            # Log suspicious activity if multiple faces are detected
            self.suspicious_logs.append("Multiple faces detected.")
            return True, "Multiple faces detected."

        # Example: Add noise detection here
        # noise_detected = False  # Replace with actual noise detection logic
        # if noise_detected:
        #     self.suspicious_logs.append("High noise levels detected.")
        #     return True, "High noise levels detected."

        return False, "No suspicious activity detected."

    def get_logs(self):
        """Retrieve the logs of suspicious activities."""
        return self.suspicious_logs