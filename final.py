import cv2
import face_recognition
import numpy as np
import sqlite3
from twilio.rest import Client
import os

# Twilio credentials
TWILIO_SID = "ACda9e22bad5a5db61801dbd82830b84c1"
TWILIO_AUTH_TOKEN = "91b74fff429e68a82625435d1aaa4b60"
TWILIO_PHONE_NUMBER = "+17754028254"
ALERT_PHONE_NUMBER = "+918591687822"

# Initialize Twilio client
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

def load_faces_from_db():
    """Loads known faces and encodings from SQLite database."""
    conn = sqlite3.connect("face_recognition.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM faces")

    known_names = []
    known_encodings = []

    for name, encoding_blob in cursor.fetchall():
        encoding = np.frombuffer(encoding_blob, dtype=np.float64)
        if encoding.shape[0] == 128:  # Ensure correct encoding size
            known_names.append(name)
            known_encodings.append(encoding)
        else:
            print(f"⚠ Skipping {name}, incorrect encoding size")

    conn.close()
    return known_encodings, known_names

# Load known faces
known_encodings, known_names = load_faces_from_db()

# Video source
video_path = "finalvideo.mp4"

if not os.path.exists(video_path):
    print(f"⚠ Error: Video file '{video_path}' not found.")
    exit()

capture = cv2.VideoCapture(video_path)

if not capture.isOpened():
    print("⚠ Error: Could not open video file.")
    exit()

unknown_faces_detected = set()
frame_skip = 2  # Process every 2nd frame
frame_count = 0

while True:
    ret, frame = capture.read()
    if not ret:
        print("⚠ End of video reached or could not retrieve frame.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frames to improve speed

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    print(f"Faces detected: {len(face_locations)}")  # Debugging output

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(distances) if len(distances) > 0 else None
        name = "⚠ Unknown Person ⚠"

        if best_match_index is not None and distances[best_match_index] < 0.6:
            name = known_names[best_match_index]
        else:
            # Generate a unique ID for unknown faces
            unknown_id = tuple(face_encoding[:5])
            if unknown_id not in unknown_faces_detected:
                unknown_faces_detected.add(unknown_id)

                # Send SMS alert for unknown face
                try:
                    message = client.messages.create(
                        body="🚨 ALERT: Unknown person detected! 🚨",
                        from_=TWILIO_PHONE_NUMBER,
                        to=ALERT_PHONE_NUMBER
                    )
                    print("📩 SMS Alert Sent!")
                except Exception as sms_error:
                    print(f"⚠ Twilio Error: {sms_error}")

        # Scale back coordinates to match original frame size
        top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2

        # Draw bounding box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display the video with detected faces
    cv2.imshow("AI Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

capture.release()
cv2.destroyAllWindows()
