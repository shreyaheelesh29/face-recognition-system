import cv2
import face_recognition
import numpy as np
import sqlite3

def load_faces_from_db():
    conn = sqlite3.connect("face_recognition.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM faces")

    known_names = []
    known_encodings = []
    for name, encoding_blob in cursor.fetchall():
        encoding = np.frombuffer(encoding_blob, dtype=np.float64)
        known_names.append(name)
        known_encodings.append(encoding)

    conn.close()
    return known_encodings, known_names

known_encodings, known_names = load_faces_from_db()

video_path = "videous.mp4"
capture = cv2.VideoCapture(video_path)

if not capture.isOpened():
    print("Error: Cannot open video file.")
    exit()

frame_skip = 2  # Process every 2nd frame
frame_count = 0

while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frames to improve speed

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Reduce resolution
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)  # Use "cnn" if on GPU
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(distances)
        name = "Unknown"

        if distances[best_match_index] < 0.6:  
            name = known_names[best_match_index]

        # Draw bounding box and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
