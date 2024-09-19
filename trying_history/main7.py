import cv2
from main2 import multi_scale_template_matching

# Load the screenshot and template (icon) images in grayscale
screenshot = cv2.imread('screenshot.png', 0)
template = cv2.imread('icon2.png', 0)

# Define scale range from 0.2 times smaller to 5 times larger
scale_ranges = (0.2, 5)

# Run the multi-scale template matching
multi_scale_template_matching(screenshot, template, scale_ranges)
