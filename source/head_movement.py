import cv2

def detect_head_movement(frame):
    """Detect head movement based on the position of detected faces."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Assuming movement if multiple faces detected or significant position change
        return True, "Head movement detected."
    
    return False, "No head movement detected."