import cv2
import numpy as np

def multi_scale_template_matching(screenshot, template, scale_ranges, threshold=0.5, overlap_thresh=0.9):
    # Check if images are loaded properly
    if screenshot is None:
        print("Error loading screenshot image.")
        return
    if template is None:
        print("Error loading template image.")
        return

    # Get dimensions of the screenshot and template
    screenshot_height, screenshot_width = screenshot.shape[:2]
    icon_height, icon_width = template.shape[:2]

    # Print the size of the screenshot and the icon
    print(f"Screenshot Size: Width = {screenshot_width}, Height = {screenshot_height}")
    print(f"Icon Size: Width = {icon_width}, Height = {icon_height}")

    # List to hold all matches
    matches = []

    # Precompute scales
    scales = []
    scale = scale_ranges[0]
    while scale < scale_ranges[1]:
        scales.append(scale)
        # Determine the step based on the current scale
        if scale < 1:
            step = 0.05  # smaller steps for scales less than 1
        elif scale < 2:
            step = 0.1  # medium steps for scales between 1 and 2
        else:
            step = 0.2  # larger steps for scales greater than 2
        scale += step

    # Iterate over the precomputed scales
    for scale in scales:
        # Resize the template based on the current scale
        resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        r_height, r_width = resized_template.shape[:2]

        # Check if the resized template is larger than the screenshot
        if r_height > screenshot_height or r_width > screenshot_width:
            continue  # Skip this scale

        # Apply template matching at the current scale
        res = cv2.matchTemplate(screenshot, resized_template, cv2.TM_CCOEFF_NORMED)

        # Find locations where matches exceed the threshold
        loc = np.where(res >= threshold)
        scores = res[loc]
        for pt, score in zip(zip(*loc[::-1]), scores):  # Switch to (x, y) coordinates
            # Store scale, top-left corner, bottom-right corner, rescaled icon size, and matching score
            match_info = {
                'scale': scale,
                'top_left': (pt[0], pt[1]),
                'bottom_right': (pt[0] + r_width, pt[1] + r_height),
                'resized_icon_size': (r_width, r_height),
                'score': score
            }
            matches.append(match_info)

    # Apply Non-Maximum Suppression to remove overlapping boxes
    if matches:
        # Prepare data for NMS
        boxes = []
        scores = []
        for match in matches:
            x1, y1 = match['top_left']
            x2, y2 = match['bottom_right']
            boxes.append([x1, y1, x2, y2])
            scores.append(match['score'])

        # Convert to numpy arrays
        boxes = np.array(boxes)
        scores = np.array(scores)

        # Perform NMS
        indices = non_max_suppression(boxes, scores, overlap_thresh)
        final_matches = [matches[i] for i in indices]

        # Print consolidated matches
        print(f"Total Matches Found after NMS: {len(final_matches)}")
        for match in final_matches:
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

def non_max_suppression(boxes, scores, overlapThresh):
    # If there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # Compute the area of the bounding boxes and sort the bounding boxes by the score
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)[::-1]  # Sort in descending order

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the index with highest score
        i = idxs[0]
        pick.append(i)

        # Find the intersection areas
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[1:]]

        # Delete all indexes from the index list that have overlap greater than the threshold
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlapThresh)[0] + 1)))

    # Return only the bounding boxes that were picked
    return pick

# Load the screenshot and template (icon) images in grayscale
screenshot = cv2.imread('screenshot.png', 0)
template = cv2.imread('icon.png', 0)

# Define scale range from 0.2 times smaller to 5 times larger
scale_ranges = (0.2, 5)

# Run the multi-scale template matching with Non-Maximum Suppression
multi_scale_template_matching(screenshot, template, scale_ranges, threshold=0.5, overlap_thresh=0.9)
