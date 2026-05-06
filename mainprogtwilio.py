import cv2
import face_recognition
import numpy as np
from twilio.rest import Client

# Twilio credentials (Replace with your actual details)
TWILIO_SID = "ACda9e22bad5a5db61801dbd82830b84c1"
TWILIO_AUTH_TOKEN = "91b74fff429e68a82625435d1aaa4b60"
TWILIO_PHONE_NUMBER = "+17754028254"  # Your Twilio number
ALERT_PHONE_NUMBER = "+918591687822"  # Your phone number

# Initialize Twilio client
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Load known faces
known_face_encodings = []
known_face_names = []

# Load multiple face images dynamically
face_data = {
    "divya": "divya.jpg",
    "aaditi": "aaditi.jpg",
    "shreya": "shreya.jpg",
     "tejsi": "tejsi.jpg",
    "tanishka": "tanishka.jpg",
     "avni": "avni.jpg",
    "nandini": "nandini.jpg",
    "vaishnavi": "vaishnavi.jpg"
}

for name, filename in face_data.items():
    try:
        image = face_recognition.load_image_file(filename)
        encodings = face_recognition.face_encodings(image)

        if encodings:  # Check if encoding exists
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
        else:
            print(f"⚠ Warning: No face found in {filename}")

    except Exception as e:
        print(f"⚠ Error loading {filename}: {e}")

print(f"✅ Loaded {len(known_face_encodings)} known faces.")

# Open webcam
video_capture = cv2.VideoCapture(0)

unknown_faces_detected = set()  # Store already detected unknown faces

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("⚠ Error: Couldn't access webcam.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Speed optimization
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "⚠ Unknown Person ⚠"

        if any(matches):
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            # Avoid sending multiple SMS alerts for the same unknown face
            unknown_id = tuple(face_encoding)
            if unknown_id not in unknown_faces_detected:
                unknown_faces_detected.add(unknown_id)
                try:
                    message = client.messages.create(
                        body="🚨 ALERT: Unknown person detected! 🚨",
                        from_=TWILIO_PHONE_NUMBER,
                        to=ALERT_PHONE_NUMBER
                    )
                    print("📩 SMS Alert Sent!")
                except Exception as sms_error:
                    print(f"⚠ Twilio Error: {sms_error}")

        # Scale back face coordinates to original frame size
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

        # Draw rectangle & label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("AI Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
