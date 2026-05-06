import cv2
import numpy as np

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture (0 indicates the default webcam)
video_capture = cv2.VideoCapture(0)

# Load the reference image and convert to grayscale
reference_image = cv2.imread('reference.jpg')
grayscale_ref_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Detect faces in the reference image
ref_faces = face_cascade.detectMultiScale(grayscale_ref_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

# Draw rectangles around faces in the reference image
for (x, y, w, h) in ref_faces:
    cv2.rectangle(reference_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

while True:
    # Capture frame-by-frame from webcam
    ret, frame = video_capture.read()
    
    if not ret:
        print("Failed to capture video feed")
        break
   
    # Convert the webcam frame to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the webcam frame
    faces = face_cascade.detectMultiScale(grayscale_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Draw rectangles around faces in the webcam feed
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Combine the reference image and webcam feed side-by-side
    combined_image = np.hstack((reference_image, frame))
    
    # Display the resulting frame with detected faces from both sources
    cv2.imshow('Face Detection: Reference Image and Webcam', combined_image)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
video_capture.release()
cv2.destroyAllWindows()