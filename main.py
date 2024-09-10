import cv2
import numpy as np

# Load the screenshot and template (icon) images
screenshot = cv2.imread('screenshot.png', 0)  # Load in grayscale
template = cv2.imread('icon.png', 0)  # Load in grayscale

# Get dimensions of the screenshot and template (icon)
screenshot_height, screenshot_width = screenshot.shape
icon_height, icon_width = template.shape

# Print the size of the screenshot and the icon
print(f"Screenshot Size: Width = {screenshot_width}, Height = {screenshot_height}")
print(f"Icon Size: Width = {icon_width}, Height = {icon_height}")

# Apply template matching
res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)

# Set the threshold for matching (adjust based on your needs)
threshold = 0.8  
loc = np.where(res >= threshold)

# Print the locations where the icon is found in the screenshot
print("Locations of the icon in the screenshot:")
for pt in zip(*loc[::-1]):  # Reversing the order to (x, y)
    print(f"Top-left corner at: (X: {pt[0]}, Y: {pt[1]})")
    # Draw rectangles around detected icons
    cv2.rectangle(screenshot, pt, (pt[0] + icon_width, pt[1] + icon_height), (0, 255, 255), 2)

# Display the result with detected icons
cv2.imshow('Detected Icons', screenshot)
cv2.waitKey(0)
cv2.destroyAllWindows()
