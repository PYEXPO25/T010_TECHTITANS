import cv2
import dlib

# Load Dlib’s pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download required

def detect_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = landmarks.part(36).x, landmarks.part(36).y  # Left eye corner
        right_eye = landmarks.part(45).x, landmarks.part(45).y  # Right eye corner
        return True  # Eyes detected

    return False  # No eyes detected (could mean closed eyes)
