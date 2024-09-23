import cv2
import numpy as np

def multi_scale_template_matching(screenshot, template, scale_ranges, threshold=0.5):
    # Get dimensions of the screenshot and template
    screenshot_height, screenshot_width = screenshot.shape[:2]
    icon_height, icon_width = template.shape[:2]

    # Print the size of the screenshot and the icon
    print(f"Screenshot Size: Width = {screenshot_width}, Height = {screenshot_height}")
    print(f"Icon Size: Width = {icon_width}, Height = {icon_height}")

    # List to hold all matches
    matches = []

    # Iterate over scales with variable steps based on the scale
    scale = scale_ranges[0]
    while scale < scale_ranges[1]:
        # Determine the step based on the current scale
        if scale < 1:
            step = 0.05  # smaller steps for scales less than 1
        elif scale < 2:
            step = 0.1  # medium steps for scales between 1 and 2
        else:
            step = 0.2  # larger steps for scales greater than 2

        # Resize the template based on the current scale
        resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        r_height, r_width = resized_template.shape[:2]

        # Apply template matching at the current scale
        res = cv2.matchTemplate(screenshot, resized_template, cv2.TM_CCOEFF_NORMED)

        # Find locations where matches exceed the threshold
        loc = np.where(res >= threshold)
        if len(loc[0]) > 0:
            # Collect match information
            for pt in zip(*loc[::-1]):  # Switch to (x, y) coordinates
                # Store scale, top-left corner, bottom-right corner, rescaled icon size
                match_info = {
                    'scale': scale,
                    'top_left': (pt[0], pt[1]),
                    'bottom_right': (pt[0] + r_width, pt[1] + r_height),
                    'resized_icon_size': (r_width, r_height)
                }
                matches.append(match_info)

        scale += step

    # After processing all scales, print all matches
    if matches:
        print(f"Total Matches Found: {len(matches)}")
        for match in matches:
            scale = match['scale']
            top_left = match['top_left']
            bottom_right = match['bottom_right']
            resized_icon_size = match['resized_icon_size']
            print(f"\nScale: {scale:.2f}")
            print(f"Resized Icon Size: Width = {resized_icon_size[0]}, Height = {resized_icon_size[1]}")
            print(f"Match Location:")
            print(f"  Top-left:     (X: {top_left[0]}, Y: {top_left[1]})")
            print(f"  Bottom-right: (X: {bottom_right[0]}, Y: {bottom_right[1]})")
    else:
        print("No matches found.")

# Load the screenshot and template (icon) images in grayscale
screenshot = cv2.imread('screenshot.png', 0)
template = cv2.imread('icon.png', 0)

# Check if images are loaded properly
if screenshot is None:
    print("Error loading screenshot image.")
if template is None:
    print("Error loading template image.")

# Define scale range from 0.2 times smaller to 5 times larger
scale_ranges = (0.2, 5)

# Run the multi-scale template matching
multi_scale_template_matching(screenshot, template, scale_ranges)
