import cv2
import numpy as np

def differentiate(img):
    # Calculate horizontal and vertical gradients using simple differences
    horizontal_gradient = cv2.absdiff(img[:, 1:], img[:, :-1])
    vertical_gradient = cv2.absdiff(img[1:, :], img[:-1, :])

    # Pad zeros to maintain the original shape
    horizontal_gradient = np.pad(horizontal_gradient, ((0, 0), (1, 0)), 'constant')
    vertical_gradient = np.pad(vertical_gradient, ((1, 0), (0, 0)), 'constant')

    # Combine gradients
    return cv2.addWeighted(horizontal_gradient, 0.5, vertical_gradient, 0.5, 0)

# Load the screenshot and template (icon) images in grayscale
screenshot = cv2.imread('screenshot.png', 0)
template = cv2.imread('icon.png', 0)

# Apply the differentiation function to both images
differentiated_screenshot = differentiate(screenshot)
differentiated_template = differentiate(template)

# Display the original and differentiated images using OpenCV
cv2.imshow('Original Screenshot', screenshot)
cv2.imshow('Differentiated Screenshot', differentiated_screenshot)
cv2.imshow('Original Icon', template)
cv2.imshow('Differentiated Icon', differentiated_template)

cv2.waitKey(0)
cv2.destroyAllWindows()
