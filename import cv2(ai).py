import cv2
import mediapipe as mp
import numpy as np
import sys

# Increase recursion limit as a safeguard (temporary fix)
sys.setrecursionlimit(5000)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Initialize OpenCV Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to capture frame.")
        break

    # Ensure the frame is a valid NumPy array
    if not isinstance(frame, np.ndarray):
        print("Error: Frame is not a valid NumPy array.")
        break

    # Convert the image to RGB (MediaPipe requirement)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = face_detection.process(rgb_frame)

    # Draw face detections
    if results.detections:
        h, w, _ = frame.shape
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            # Ensure bounding box values are within frame limits
            x = max(0, min(int(bbox.xmin * w), w - 1))
            y = max(0, min(int(bbox.ymin * h), h - 1))
            width = max(1, min(int(bbox.width * w), w - x))
            height = max(1, min(int(bbox.height * h), h - y))

            # Draw rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("AI Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()