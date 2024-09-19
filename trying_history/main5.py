import cv2
import numpy as np

def pyramid_template_matching(screenshot, template, max_levels, threshold=0.8):
    for level in range(max_levels):
        # Downscale images
        scaled_screenshot = cv2.pyrDown(screenshot)
        scaled_template = cv2.pyrDown(template)

        # Update the screenshot and template
        screenshot = scaled_screenshot if scaled_screenshot.shape[0] > template.shape[0] else screenshot
        template = scaled_template if scaled_template.shape[0] > 0 else template

        res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(screenshot, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0, 255, 0), 2)

    cv2.imshow('Detected Icons', screenshot)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load images
screenshot = cv2.imread('screenshot.png', 0)
template = cv2.imread('icon.png', 0)
max_levels = 3  # Number of pyramid levels

# Run the pyramid matching
pyramid_template_matching(screenshot, template, max_levels)
