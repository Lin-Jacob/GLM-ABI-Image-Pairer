import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread(
    'images\G16_ABI_B03_s20233172000203_e20233172009511_x00000y-02400.nc.png')
gray_image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)

# Intialize SIFT algorithm
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray_image, None)

image_with_keypoints = cv2.drawKeypoints(
    image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the image with keypoints
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('SIFT Keypoints')
plt.show()

