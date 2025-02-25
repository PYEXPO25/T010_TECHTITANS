import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh model
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize the Face Mesh model
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect_eye_movement(frame):
    """Detect if there is significant eye movement in the frame using MediaPipe."""
    # Convert the image to RGB as MediaPipe expects it in RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and get the facial landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get the coordinates of the left and right eyes
            left_eye = face_landmarks.landmark[33]  # Left eye landmark
            right_eye = face_landmarks.landmark[133]  # Right eye landmark

            # Calculate horizontal distance between the eyes
            eye_distance = abs(left_eye.x - right_eye.x)

            # Example logic: check if eyes are significantly shifted (can be extended)
            eye_movement = False
            if eye_distance > 0.03:  # Arbitrary threshold for movement
                eye_movement = True

            if eye_movement:
                return True, "Eye movement detected."

    return False, "No significant eye movement detected."

# Sample code to run the function on video
cap = cv2.VideoCapture(0)  # Open webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Call the eye movement detection function
    eye_movement, message = detect_eye_movement(frame)
    
    # Display the result
    cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Eye Movement Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
