import cv2
import face_recognition
import numpy as np
import os

# Directory to store known face
KNOWN_FACE_DIR = "known_faces"
if not os.path.exists(KNOWN_FACE_DIR):
    os.makedirs(KNOWN_FACE_DIR)

# Load a stored face image if it exists
known_face_encodings = []
known_face_names = []

# Check if a stored face exists
for file in os.listdir(KNOWN_FACE_DIR):
    img_path = os.path.join(KNOWN_FACE_DIR, file)
    img = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(img)[0]  # Encode stored face
    known_face_encodings.append(encoding)
    known_face_names.append(file.split(".")[0])  # Use filename as identity

# Initialize Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            matched_index = matches.index(True)
            name = known_face_names[matched_index]
            print("ALERT: Found here ->", name)  # Alert message
        else:
            # If no match, save the detected face
            (top, right, bottom, left) = face_location
            face_image = frame[top:bottom, left:right]
            cv2.imwrite(os.path.join(KNOWN_FACE_DIR, "face.jpg"), face_image)
            print("New face stored as face.jpg")
            break  # Stop further execution so face isn't stored multiple times

        # Draw rectangle and name
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()