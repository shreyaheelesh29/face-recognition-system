import os
import cv2
import numpy as np
import face_recognition
from twilio.rest import Client

# Fetch Twilio credentials from environment variables
TWILIO_ACCOUNT_SID = os.getenv("ACda9e22bad5a5db61801dbd82830b84c1")
TWILIO_AUTH_TOKEN = os.getenv("91b74fff429e68a82625435d1aaa4b60")
TWILIO_PHONE_NUMBER = os.getenv("+17754028254")
RECIPIENT_PHONE_NUMBER = os.getenv("8591687822")

# Debugging check for credentials
if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    print("Error: Twilio credentials not set!")
    exit(1)
else:
    print("Twilio credentials loaded successfully.")

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Load known faces (dummy example, replace with your database logic)
known_face_encodings = []
known_face_names = []

# Initialize webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not access the webcam.")
    exit(1)

unknown_alert_sent = False  # Prevent duplicate alerts

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "⚠ Unknown Person Detected!"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            unknown_alert_sent = False  # Reset alert if known person is detected
        else:
            if not unknown_alert_sent:
                try:
                    print("Sending SMS Alert...")
                    message = client.messages.create(
                        body="⚠ Alert! Unknown face detected. Please check the camera feed.",
                        from_=+17754028254,
                        to=8591687822
                    )
                    print(f"SMS Alert Sent: {message.sid}")
                except Exception as e:
                    print(f"Error sending SMS: {e}")
                unknown_alert_sent = True

        # Draw rectangle around the face
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show video feed
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()