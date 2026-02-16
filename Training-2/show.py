# import numpy as np

# keypoints = np.load(r"C:\Rishika\MajorProject_1\Training-2\keypoints_2D\2\0.npy")
# print(keypoints)
# print(keypoints.shape)

# import cv2

# image = cv2.imread(r"C:\Rishika\MajorProject_1\Indian\2\0.jpg")

# if image is None:
#     print("Image not loaded")
#     exit()

# h, w, _ = image.shape

import numpy as np
import cv2

# Load keypoints
keypoints = np.load(r"C:\Rishika\MajorProject_1\Training-2\keypoints_2D\2\0.npy")
keypoints = keypoints.reshape(21, 2)

# Load image
image = cv2.imread(r"C:\Rishika\MajorProject_1\Indian\2\0.jpg")

if image is None:
    print("Image not loaded")
    exit()

h, w, _ = image.shape

# Normalize keypoints to 0-1
min_vals = keypoints.min(axis=0)
max_vals = keypoints.max(axis=0)

normalized = (keypoints - min_vals) / (max_vals - min_vals)

# Draw
for (x, y) in normalized:
    x_pixel = int(x * w)
    y_pixel = int(y * h)
    cv2.circle(image, (x_pixel, y_pixel), 6, (0,255,0), -1)

cv2.imshow("Keypoints", image)
cv2.waitKey(0)
cv2.destroyAllWindows()