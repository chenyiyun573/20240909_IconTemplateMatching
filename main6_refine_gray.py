import cv2
import numpy as np

def preprocess_image(img):
    # Apply Histogram Equalization
    img_eq = cv2.equalizeHist(img)
    # Apply Adaptive Thresholding
    img_thresh = cv2.adaptiveThreshold(img_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    return img_thresh

def multi_scale_template_matching(screenshot, template, scale_ranges, threshold=0.5):
    # Preprocess images
    processed_screenshot = preprocess_image(screenshot)
    processed_template = preprocess_image(template)

    # Get dimensions of the images
    screenshot_height, screenshot_width = processed_screenshot.shape
    icon_height, icon_width = processed_template.shape

    print(f"Screenshot Size: Width = {screenshot_width}, Height = {screenshot_height}")
    print(f"Icon Size: Width = {icon_width}, Height = {icon_height}")

    scale = scale_ranges[0]
    while scale < scale_ranges[1]:
        if scale < 1:
            step = 0.05
        elif scale < 2:
            step = 0.1
        else:
            step = 0.2

        resized_template = cv2.resize(processed_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        r_height, r_width = resized_template.shape[:2]

        # Apply template matching
        res = cv2.matchTemplate(processed_screenshot, resized_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        print(f"Checking scale {scale:.2f}: Found {len(loc[0])} matches.")

        for pt in zip(*loc[::-1]):
            cv2.rectangle(screenshot, pt, (pt[0] + r_width, pt[1] + r_height), (0, 255, 0), 2)
            print(f"Top-left corner at: (X: {pt[0]}, Y: {pt[1]})")

        scale += step

    cv2.imshow('Detected Icons', screenshot)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load images in grayscale
screenshot = cv2.imread('screenshot.png', 0)
template = cv2.imread('icon.png', 0)

# Define scale ranges
scale_ranges = (0.2, 5)

# Run the multi-scale template matching
multi_scale_template_matching(screenshot, template, scale_ranges)
