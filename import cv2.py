import cv2

def detect_faces(girl.jpg):
    try:
        # Load the Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        if face_cascade.empty():
            raise IOError("Haar Cascade XML file not found.")

        # Load the image
        image = cv2.imread(girl.jpg)

        if image is None:
            raise FileNotFoundError("Image file not found or could not be loaded.")

        # Convert the image to grayscale for detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the result
        cv2.imshow('Face Detection', image)

        print(f"Number of faces detected: {len(faces)}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except cv2.error as e:
        print(f"OpenCV Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
girl.jpg = "path_to_image.jpg"  # Replace with the path to your image
detect_faces(girl.jpg)