import cv2
import numpy as np

def feature_based_matching(screenshot, template, scale_ranges):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors for the template
    kp_template, des_template = orb.detectAndCompute(template, None)

    scale = scale_ranges[0]
    while scale < scale_ranges[1]:
        if scale < 1:
            step = 0.05
        elif scale < 2:
            step = 0.1
        else:
            step = 0.2

        # Resize the template
        resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        kp_resized_template, des_resized_template = orb.detectAndCompute(resized_template, None)

        # Detect keypoints and compute descriptors for the screenshot
        kp_screenshot, des_screenshot = orb.detectAndCompute(screenshot, None)

        # Create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if des_resized_template is not None and des_screenshot is not None:
            # Match descriptors
            matches = bf.match(des_resized_template, des_screenshot)
            matches = sorted(matches, key=lambda x: x.distance)

            print(f"Checking scale {scale:.2f}: Found {len(matches)} matches.")

            if len(matches) > 10:
                # Draw matches
                result = cv2.drawMatches(resized_template, kp_resized_template,
                                         screenshot, kp_screenshot, matches[:10], None, flags=2)
                cv2.imshow('Matches', result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print(f"Checking scale {scale:.2f}: Not enough descriptors.")

        scale += step

# Load images in grayscale
screenshot = cv2.imread('screenshot.png', 0)
template = cv2.imread('icon.png', 0)

# Define scale ranges
scale_ranges = (0.2, 5)

# Run feature-based matching
feature_based_matching(screenshot, template, scale_ranges)
