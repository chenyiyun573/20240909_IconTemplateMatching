import cv2
import numpy as np

def multi_scale_template_matching(screenshot, template, scale_ranges, step=0.1, threshold=0.5):
    # Get dimensions of the screenshot and template
    screenshot_height, screenshot_width = screenshot.shape
    icon_height, icon_width = template.shape

    # Print the size of the screenshot and the icon
    print(f"Screenshot Size: Width = {screenshot_width}, Height = {screenshot_height}")
    print(f"Icon Size: Width = {icon_width}, Height = {icon_height}")

    # Iterate over scale ranges with a defined step
    for scale in np.arange(scale_ranges[0], scale_ranges[1], step):
        resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)  # Resizing template
        r_height, r_width = resized_template.shape

        # Apply template matching at the current scale
        res = cv2.matchTemplate(screenshot, resized_template, cv2.TM_CCOEFF_NORMED)

        # Find locations where matches exceed the threshold
        loc = np.where(res >= threshold)
        print(f"Checking scale {scale:.2f}: Found {len(loc[0])} matches.")

        # Draw rectangles for all matches found at this scale
        for pt in zip(*loc[::-1]):  # Switch to (x, y) coordinates
            cv2.rectangle(screenshot, pt, (pt[0] + r_width, pt[1] + r_height), (0, 255, 0), 2)
            print(f"Top-left corner at: (X: {pt[0]}, Y: {pt[1]})")

    # Display the result with detected icons
    cv2.imshow('Detected Icons', screenshot)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Load the screenshot and template (icon) images in grayscale
screenshot = cv2.imread('screenshot.png', 0)
template = cv2.imread('icon.png', 0)

# Define scale range from 0.2 times smaller to 5 times larger
scale_ranges = (0.2, 5)

# Run the multi-scale template matching
multi_scale_template_matching(screenshot, template, scale_ranges)
