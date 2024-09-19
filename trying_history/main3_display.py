import cv2
import numpy as np

# Load the screenshot and template (icon) images in grayscale
screenshot = cv2.imread('screenshot.png', 0)
template = cv2.imread('icon.png', 0)

# Convert images to edges using Canny edge detector
edge_screenshot = cv2.Canny(screenshot, 50, 200)
edge_template = cv2.Canny(template, 50, 200)

# Display the original and edge-detected images
cv2.imshow('Original Screenshot', screenshot)
cv2.imshow('Edge-Detected Screenshot', edge_screenshot)
cv2.imshow('Original Icon', template)
cv2.imshow('Edge-Detected Icon', edge_template)

cv2.waitKey(0)
cv2.destroyAllWindows()
