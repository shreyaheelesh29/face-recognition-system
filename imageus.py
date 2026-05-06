import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read the image
image = cv2.imread('imageus.jpg')
image = cv2.imread('imageus1.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.1,  # Adjust the scale factor for accuracy
    minNeighbors=5,   # Adjust the number of neighbors for accuracy
    minSize=(30, 30)  # Set minimum size for detected faces
)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the output image with detected faces
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()