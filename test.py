import cv2
import numpy as np

# Load the two images
image1 = cv2.imread('images\G16_GLM_2023_11_13_200402_x-00100y00000.nc.png')
image2 = cv2.imread(
    'images\G16_ABI_B03_s20233172000203_e20233172009511_x00000y-02400.nc.png')

# Get the height of the taller image
height = max(image1.shape[0], image2.shape[0])

# Create a blank canvas with the height of the taller image and the combined width of both images
combined_image = np.zeros(
    (height, image1.shape[1] + image2.shape[1], 3), dtype=np.uint8)

# Place the first image on the canvas
combined_image[:image1.shape[0], :image1.shape[1]] = image1

# Place the second image on the canvas
combined_image[:image2.shape[0], image1.shape[1]
    :image1.shape[1] + image2.shape[1]] = image2

# Display the combined image
cv2.imshow('Combined Image', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
