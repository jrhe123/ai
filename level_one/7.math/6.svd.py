import cv2
import numpy as np

# Read the image
image = cv2.imread('robot.bmp', 0)  # Reads the image in grayscale

# Perform SVD on the image
U, S, V = np.linalg.svd(image.astype(np.float64), full_matrices=False)

# Define the number of singular values to retain
k = 10
s_k = np.diag(S[:k])  # Keep only the top k singular values

# Reconstruct the image using the top k singular values
compressed_image = np.dot(U[:, :k], np.dot(s_k, V[:k, :]))
compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)  # Clip values and convert to uint8

# Display the original and compressed images
cv2.imshow('Original Image', image)
cv2.imshow('Compressed Image', compressed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
