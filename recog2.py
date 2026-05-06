import cv2
import numpy as np

def face_recognition_opencv(known_images_paths, unknown_image_path):
    try:
        # Step 1: Load known images and compute their features
        orb = cv2.ORB_create(nfeatures=1000)  # Initialize ORB detector
        known_features = []  # Store known descriptors
        known_names = []     # Store corresponding names

        for name, image_path in known_images_paths.items():
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error loading {image_path}")
                continue
            keypoints, descriptors = orb.detectAndCompute(image, None)
            if descriptors is not None:
                known_features.append((descriptors, name))
                known_names.append(name)
            else:
                print(f"Warning: No descriptors found in {image_path}, skipping.")

        # Step 2: Load the unknown image
        unknown_image = cv2.imread(unknown_image_path)
        if unknown_image is None:
            raise Exception(f"Error loading {unknown_image_path}")
        gray_unknown = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2GRAY)

        # Step 3: Detect faces in the unknown image
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray_unknown, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        # If faces are detected, pick the largest one
        if len(faces) == 0:
            print("No faces found in the image.")
            return

        # Sort faces by area (largest first)
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        # Focus only on the largest face
        (x, y, w, h) = faces[0]

        face_roi = gray_unknown[y:y+h, x:x+w]
        keypoints, descriptors = orb.detectAndCompute(face_roi, None)

        name = "Unknown"  # Default name if no match is found
        if descriptors is not None:
            bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            best_score = 0
            for known_desc, known_name in known_features:
                matches = bf_matcher.knnMatch(known_desc, descriptors, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

                if len(good_matches) > best_score:
                    best_score = len(good_matches)
                    name = known_name if best_score > 10 else "Unknown"

        # Draw the largest face rectangle and label
        cv2.rectangle(unknown_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(unknown_image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Step 5: Display the final image
        cv2.imshow("Face Recognition", unknown_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

# Known images with names
known_images = {
    "Person1": "aaditi.jpg",  # Replace with actual paths to known images
    "Person2": "divya.jpg"
}

# Path to the unknown image
unknown_image = "imageus.jpg"  # Replace with the path to your unknown image

# Call the function
face_recognition_opencv(known_images, unknown_image)