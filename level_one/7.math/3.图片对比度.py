import cv2

image = cv2.imread('image.jpg')

increased_contrast = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

decreased_contrast = cv2.convertScaleAbs(image, alpha=0.5, beta=0)

cv2.imshow('Original Image', image)
cv2.imshow('Increased Contrast', increased_contrast)
cv2.imshow('Decreased Contrast', decreased_contrast)

cv2.waitKey(0)