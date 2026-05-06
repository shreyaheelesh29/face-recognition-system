import cv2
import numpy as np
import os

def recognize_faces_opencv(known_images, unknown_image_path):
    try:
        # Step 1: Load known images and compute features
        known_features = []
        known_names = []

        orb = cv2.ORB_create()  # ORB feature detector

        for name, image_path in known_images.items():
            # Load the known image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error loading {image_path}")
                continue

            # Detect and compute features
            keypoints, descriptors = orb.detectAndCompute(image, None)
            known_features.append(descriptors)
            known_names.append(name)

        # Step 2: Load unknown image
        unknown_image = cv2.imread(unknown_image_path)
        if unknown_image is None:
            raise Exception(f"Error loading {unknown_image_path}")

        gray_unknown = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2GRAY)

        # Step 3: Detect faces in the unknown image
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        face_locations = face_cascade.detectMultiScale(gray_unknown, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Step 4: For each detected face, extract and compare features
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for (x, y, w, h) in face_locations:
            # Extract face ROI
            face_roi = gray_unknown[y:y+h, x:x+w]

            # Detect and compute features for the face
            keypoints, descriptors = orb.detectAndCompute(face_roi, None)

            if descriptors is None:
                name = "Unknown"
            else:
                # Compare with known descriptors
                best_match_name = "Unknown"
                max_matches = 0

                for known_name, known_descriptor in zip(known_names, known_features):
                    if known_descriptor is None:
                        continue

                    matches = bf_matcher.match(known_descriptor, descriptors)
                    if len(matches) > max_matches:
                        max_matches = len(matches)
                        best_match_name = known_name

                name = best_match_name

            # Draw rectangle around the face and display the name
            cv2.rectangle(unknown_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(unknown_image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Step 5: Show the output
        cv2.imshow('Face Recognition', unknown_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

# Known images dictionary: {name: file_path}
known_images = {
    "Person1": "aaditi.jpg",
    "Person2": "divya.jpg"
}

# Path to the unknown image
unknown_image_path = "imageus.jpg"

# Run the face recognition
recognize_faces_opencv(known_images, unknown_image_path)