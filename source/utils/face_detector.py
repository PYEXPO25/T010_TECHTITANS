import cv2

def load_face_net():
    """Load the pre-trained face detection model."""
    # Load the Caffe model for face detection (this should be downloaded beforehand)
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    return net