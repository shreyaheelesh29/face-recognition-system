import cv2

video_path = "finalvideo.mp4"
video_capture = cv2.VideoCapture(video_path)

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    cv2.imshow("Video Playback Test", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
