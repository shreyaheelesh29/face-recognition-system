import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Replace 'your_video.mp4' with the path to your .mp4 video file
video_capture = cv2.VideoCapture('videous.mp4')

# Check if the video file was opened successfully
if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Capture frame-by-frame from the video file
    ret, frame = video_capture.read()

    # If the frame was not grabbed (end of video), break the loop
    if not ret:
        print("Failed to grab frame or end of video.")
        break
    # Convert the frame to grayscale (Haar Cascade works better on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame with detected faces
    cv2.imshow('Face Detection in Video', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()