import cv2
import numpy as np

def logarithmic_scale_template_matching(screenshot, template, min_scale, max_scale, num_scales, threshold=0.8):
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num=num_scales)
    
    # Iterate over calculated scales
    for scale in scales:
        resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)  # Resizing template
        r_height, r_width = resized_template.shape

        # Apply template matching at the current scale
        res = cv2.matchTemplate(screenshot, resized_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        # Draw rectangles for all matches found at this scale
        for pt in zip(*loc[::-1]):
            cv2.rectangle(screenshot, pt, (pt[0] + r_width, pt[1] + r_height), (0, 255, 0), 2)
            print(f"Scale: {scale:.2f}, Location: (X: {pt[0]}, Y: {pt[1]})")

    cv2.imshow('Detected Icons', screenshot)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load images
screenshot = cv2.imread('screenshot.png', 0)
template = cv2.imread('icon.png', 0)

# Settings for logarithmic scaling
min_scale = 0.2
max_scale = 5
num_scales = 20

# Run the matching
logarithmic_scale_template_matching(screenshot, template, min_scale, max_scale, num_scales)
