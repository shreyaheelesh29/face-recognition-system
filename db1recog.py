import sqlite3
import face_recognition
import numpy as np
import cv2

# Connect to database
conn = sqlite3.connect("face_recognition.db")
cursor = conn.cursor()

# Load known faces from the database
cursor.execute("SELECT name, encoding FROM faces")
known_faces = cursor.fetchall()

known_face_encodings = []
known_face_names = []

# Convert the encoding from BLOB to numpy array
for name, encoding in known_faces:
    face_encoding = np.frombuffer(encoding, dtype=np.float64)  # Convert from BLOB to array
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)

conn.close()

# Initialize webcam
video_capture = cv2.VideoCapture(0)


while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Speed optimization
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare detected face with known faces from the database
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "⚠ Unknown person - Be Alert!"

        # If a match is found
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw rectangle around the face
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4  # Resize back to original size
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show video feed
    cv2.imshow("Face Recognition", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break