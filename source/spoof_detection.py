import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load Real & Fake Face Images (Assume you have a dataset)
real_images = ["real1.jpg", "real2.jpg"]  # Replace with actual images
fake_images = ["fake1.jpg", "fake2.jpg"]  # Printed photo or digital display

data = []
labels = []

def extract_lbp_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (100, 100))
    lbp = local_binary_pattern(image, 24, 8, method="uniform")  # 24 points, radius 8
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float")
    hist /= hist.sum()  # Normalize
    return hist

# Extract LBP features from real & fake faces
for img in real_images:
    data.append(extract_lbp_features(img))
    labels.append("real")

for img in fake_images:
    data.append(extract_lbp_features(img))
    labels.append("fake")

# Train an SVM classifier
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# Test on live camera feed
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lbp_features = extract_lbp_features(gray)
    prediction = model.predict([lbp_features])
    label = le.inverse_transform(prediction)[0]

    cv2.putText(frame, f"Detection: {label}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Spoof Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
