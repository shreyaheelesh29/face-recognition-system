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

# Function to add faces with optimized detection
def add_face_to_db(image_paths, name, age):
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            continue

        # Load image
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image for faster processing
        max_size = 800
        height, width = rgb_image.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            rgb_image = cv2.resize(rgb_image, (int(width * scale), int(height * scale)))

        # Try HOG first (faster)
        face_locations = face_recognition.face_locations(rgb_image, model="hog")

        # If HOG fails, try CNN
        if not face_locations:
            print(f"⚠ HOG failed for {image_path}, trying CNN...")
            face_locations = face_recognition.face_locations(rgb_image, model="cnn")

        if not face_locations:
            print(f"❌ No face found in {image_path}. Skipping...")
            continue

        # Extract face encoding
        encoding = face_recognition.face_encodings(rgb_image, known_face_locations=face_locations, num_jitters=5, model="large")

        if encoding:
            encoding_blob = np.array(encoding[0]).tobytes()  # Convert to binary
            cursor.execute("INSERT INTO faces (name, age, encoding) VALUES (?, ?, ?)", (name, age, encoding_blob))
            conn.commit()
            print(f"✅ Added {name} ({image_path}) to database.")
        else:
            print(f"⚠ No face encoding found in {image_path}.")

# Adding multiple images per person
add_face_to_db(["aaditi.jpg", "aaditi1.jpg", "aaditi2.jpg", "aaditi3.jpg"], "Aaditi", 25)
add_face_to_db(["divya.jpg", "divya1.jpg", "divya2.jpg", "divya3.jpg"], "Divya", 23)
add_face_to_db(["tejsi.jpg", "tejsi1.jpg", "tejsi2.jpg", "tejsi3.jpg"], "Tejsi", 28)
add_face_to_db(["shreya.jpg", "shreya1.jpg", "shreya2.jpg", "shreya3.jpg"], "Shreya", 27)

# Function to check database entries
def check_database():
    cursor.execute("SELECT id, name, age FROM faces")
    rows = cursor.fetchall()

    print("\n📌 Faces stored in the database:")
    for row in rows:
        print(f"ID: {row[0]}, Name: {row[1]}, Age: {row[2]}")

# Verify if the images were added successfully
check_database()

# Close connection
conn.close()