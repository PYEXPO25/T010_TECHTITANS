import cv2
from utils.face_detector import load_face_net

def detect_multiple_faces(frame):
    """Detect multiple faces in the frame."""
    # Load the face detection model from utils
    net = load_face_net()

    # Prepare the frame for face detection
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),  # Resize to 300x300 and normalize
        (104.0, 177.0, 123.0)  # Mean subtraction values for normalization
    )
    net.setInput(blob)
    detections = net.forward()

    # Count the number of faces detected
    face_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:  # Confidence threshold
            face_count += 1

    if face_count > 1:
        return True, "Multiple faces detected."
    
    return False, "No suspicious activity detected."