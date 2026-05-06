import sqlite3
import face_recognition
import numpy as np
import cv2
import os

# Connect to SQLite database
conn = sqlite3.connect("face_recognition.db")
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    encoding BLOB NOT NULL
);
""")
conn.commit()

# Function to improve image preprocessing and add faces to the database
def add_face_to_db(image_paths, name, age):
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            continue

        # Load image with OpenCV
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Resize image for better face detection
        scale_percent = 150  # Increase image size by 150%
        width = int(rgb_image.shape[1] * scale_percent / 100)
        height = int(rgb_image.shape[0] * scale_percent / 100)
        resized_image = cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_LINEAR)

        # Detect face locations first
        face_locations = face_recognition.face_locations(resized_image, model="hog")
        if not face_locations:
            print(f"No face found in {image_path}. Trying with a larger image...")
            continue  # Skip if no face found

        # Extract face encoding
        encoding = face_recognition.face_encodings(resized_image, known_face_locations=face_locations, num_jitters=10, model="large")

        if encoding:
            encoding_blob = np.array(encoding[0]).tobytes()  # Convert to binary
            cursor.execute("INSERT INTO faces (name, age, encoding) VALUES (?, ?, ?)", (name, age, encoding_blob))
            conn.commit()
            print(f"✅ Added {name} ({image_path}) to database.")
        else:
            print(f"⚠ No face encoding found in {image_path}.")

# Adding multiple images per person
add_face_to_db(["aaditi.jpg", "aaditi1.jpg", "aaditi2.jpg"], "Aaditi", 25)
add_face_to_db(["divya.jpg", "divya1.jpg", "divya2.jpg", "divya3.jpg"], "Divya", 23)
add_face_to_db(["tejsi.jpg", "tejsi1.jpg", "tejsi2.jpg"], "Tejsi", 28)
add_face_to_db(["shreya.jpg", "shreya1.jpg", "shreya2.jpg", "shreya3.jpg"], "Shreya", 27)
add_face_to_db(["vaishnavi.jpg"], "Vaishnavi", 26)


# Function to check database entries
def check_database():
    cursor.execute("SELECT id, name, age, encoding FROM faces")
    rows = cursor.fetchall()

    print("\nFaces stored in the database:")
    for row in rows:
        face_id, name, age, encoding_blob = row
        encoding = np.frombuffer(encoding_blob, dtype=np.float64)  # Convert back to numpy array
        print(f"ID: {face_id}, Name: {name}, Age: {age}, Encoding Length: {len(encoding)}")

# Verify if the images were added successfully
check_database()

# Close connection
conn.close()