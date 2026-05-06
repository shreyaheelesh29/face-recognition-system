import cv2
import face_recognition
import numpy as np
import sqlite3
import os
from twilio.rest import Client

# ✅ Twilio credentials for sending SMS alerts
TWILIO_SID = "ACda9e22bad5a5db61801dbd82830b84c1"
TWILIO_AUTH_TOKEN = "91b74fff429e68a82625435d1aaa4b60"
TWILIO_PHONE_NUMBER = "+17754028254"
ALERT_PHONE_NUMBER = "+918591687822"

# ✅ Initialize Twilio client
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# ✅ Connect to SQLite database
conn = sqlite3.connect("face_recognition.db")
cursor = conn.cursor()

# ✅ Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    encoding BLOB NOT NULL
);
""")
conn.commit()

# ✅ Function to add a face to the database
def add_face_to_db(image_paths, name, age):
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(rgb_image, (0, 0), fx=1.5, fy=1.5)

        face_locations = face_recognition.face_locations(resized_image, model="hog")
        if not face_locations:
            print(f"⚠ No face found in {image_path}. Skipping...")
            continue  

        encoding = face_recognition.face_encodings(resized_image, known_face_locations=face_locations, num_jitters=10, model="large")

        if encoding:
            encoding_blob = np.array(encoding[0]).tobytes()
            cursor.execute("INSERT INTO faces (name, age, encoding) VALUES (?, ?, ?)", (name, age, encoding_blob))
            conn.commit()
            print(f"✅ Added {name} ({image_path}) to database.")
        else:
            print(f"⚠ No encoding found in {image_path}.")

# ✅ Adding multiple images per person
add_face_to_db(["aaditi.jpg", "aaditi1.jpg"], "Aaditi", 25)
add_face_to_db(["divya.jpg", "divya1.jpg"], "Divya", 23)
add_face_to_db(["tejsi.jpg", "tejsi1.jpg"], "Tejsi", 28)
add_face_to_db(["shreya.jpg", "shreya1.jpg"], "Shreya", 27)
add_face_to_db(["vaishnavi.jpg"], "Vaishnavi", 26)

# ✅ Function to load faces from the database
def load_faces_from_db():
    cursor.execute("SELECT name, encoding FROM faces")
    known_names, known_encodings = [], []

    for name, encoding_blob in cursor.fetchall():
        encoding = np.frombuffer(encoding_blob, dtype=np.float64)
        if encoding.shape[0] == 128:
            known_names.append(name)
            known_encodings.append(encoding)
        else:
            print(f"⚠ Skipping {name}, incorrect encoding size")

    print(f"✅ Loaded known faces: {known_names}")
    return known_encodings, known_names

known_encodings, known_names = load_faces_from_db()

# ✅ Video recognition
video_path = "finalvideo.mp4"
if not os.path.exists(video_path):
    print(f"⚠ Error: Video file '{video_path}' not found.")
    exit()

capture = cv2.VideoCapture(video_path)
if not capture.isOpened():
    print("⚠ Error: Could not open video file.")
    exit()

unknown_faces_detected = set()
frame_skip = 2
frame_count = 0

while True:
    ret, frame = capture.read()
    if not ret:
        print("⚠ End of video reached or could not retrieve frame.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame, model='hog')
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    print(f"🎥 Faces detected: {len(face_locations)}")  

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(distances) if len(distances) > 0 else None
        name = "⚠ Unknown Person ⚠"

        if best_match_index is not None and distances[best_match_index] < 0.65:  
            name = known_names[best_match_index]
        else:
            unknown_id = tuple(face_encoding[:5])
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

        top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("AI Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

capture.release()
cv2.destroyAllWindows()
conn.close()