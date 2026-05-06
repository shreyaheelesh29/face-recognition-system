import cv2
image = cv2.imread('ladyface.jpg')
grayscale_image = cv2.cvtColor(image ,cv2.COLOR_BGR2GRAY)
cv2.imshow('Original Image', image)
cv2.imshow('Grayscale Image', grayscale_image)
cv2.waitKey(0)