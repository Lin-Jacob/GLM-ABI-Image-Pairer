import cv2
import numpy as np
import sift_match

image_dir = "images"
sift_matched = sift_match.sift_match(image_dir)

test = ('G16_GLM_2023_11_13_200402_x-00100y00000.nc.png', 'G16_ABI_B03_s20233172000203_e20233172009511_x00800y00000.nc.png')

print(sift_matched.keypoints[test])
'''
# Define points in the source image
good_matches = keypoints.keypoints[test].get('matched_keypoints')
#-- Localize the object
obj = np.empty((len(good_matches),2), dtype=np.float32)
scene = np.empty((len(good_matches),2), dtype=np.float32)
for i in range(len(good_matches)):
 #-- Get the keypoints from the good matches
  obj[i,0] = good_matches[i].queryIdx.pt[0]
  obj[i,1] = good_matches[i].queryIdx.pt[1]
  scene[i,0] = good_matches[i].trainIdx.pt[0]
  scene[i,1] = good_matches[i].trainIdx.pt[1]
  
print(obj)
print(scene)


# Compute the homography matrix
H, status = cv2.findHomography(src_points, dst_points)

# Read the source image
src_image = cv2.imread('source_image.jpg')

# Apply the warp transformation
dst_image = cv2.warpPerspective(src_image, H, (src_image.shape[1], src_image.shape[0]))

# Save or display the destination image
cv2.imwrite('warped_image.jpg', dst_image)
cv2.imshow('Warped Image', dst_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


