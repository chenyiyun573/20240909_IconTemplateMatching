import cv2

# Load images
screenshot = cv2.imread('screenshot.png', 0)
icon = cv2.imread('icon.png', 0)

# Create ORB detector and compute keypoints and descriptors
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(icon, None)
kp2, des2 = orb.detectAndCompute(screenshot, None)

# Match features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
result = cv2.drawMatches(icon, kp1, screenshot, kp2, matches[:10], None, flags=2)

# Display the result
cv2.imshow('Feature Matching', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
