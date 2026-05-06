import cv2

# Test loading the video
video_capture = cv2.VideoCapture('video1.mp4')

# Check if the video was opened successfully
if not video_capture.isOpened():
    print("Error: Could not open video.")
else:
    print("Video opened successfully!")

# Try reading the first frame
ret, frame = video_capture.read()
if ret:
    print("First frame loaded!")
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
else:
    print("Failed to load the first frame.")

video_capture.release()
cv2.destroyAllWindows()