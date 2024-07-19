import cv2
import numpy as np
import sift_match
import matplotlib.pyplot as plt
import textwrap
from scipy.optimize import leastsq

# Define the affine transformation
def affine_transform(params, xy):
  a0, a1, a2, b0, b1, b2 = params
  x, y = xy[:, 0], xy[:, 1]
  x_prime = a0 + a1 * x + a2 * y
  y_prime = b0 + b1 * x + b2 * y
  return np.vstack((x_prime, y_prime)).T

  # Residual function for least squares optimization
def residuals(params, xy, xy_prime):
  return (affine_transform(params, xy) - xy_prime).flatten()

class test():
  def __init__(self, dir, ransac_on : bool):
    self.image_dir = dir
    self.ransac_on = ransac_on
    self.get_assessment = {}
    self.accuracy_assessment()
  
  def accuracy_assessment(self):
    # Assuming sift_match module and images are properly set up
    sift_matched = sift_match.find_match(self.image_dir)

    for (pair, keypoints) in sift_matched.keypoints.items():
      abi_matched, glm_matched = zip(*keypoints.get("matched_keypoints"))

      src_points = np.array(glm_matched)
      dst_points = np.array(abi_matched)
      
      initial_params = np.zeros(6)
      optimized_params, _ = leastsq(residuals, initial_params, args=(src_points, dst_points))
      
      transformed_points = affine_transform(optimized_params, src_points)

      #print("Homography Matrix:\n", H)
  
      errors = np.linalg.norm(transformed_points - dst_points, axis=1)
  
      # Read the image for displaying keypoints
      compiled_image = cv2.imread(f"images/{pair[1]}")
      
      # Convert points to cv2.KeyPoint objects with larger size
      predicted_keypoints = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in transformed_points]
      abi_keypoints = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in abi_matched]
      
      # Draw keypoints on the image
      compiled_image_with_predicted = cv2.drawKeypoints(compiled_image, predicted_keypoints, None, color=(0, 0, 255))
      compiled_image_with_abi = cv2.drawKeypoints(compiled_image_with_predicted, abi_keypoints, None, color=(255, 0, 0))
      
      # Draw lines between the matched and predicted points
      for pred_kp, abi_kp in zip(predicted_keypoints, abi_keypoints):
          pred_pt = (int(pred_kp.pt[0]), int(pred_kp.pt[1]))
          abi_pt = (int(abi_kp.pt[0]), int(abi_kp.pt[1]))
          cv2.line(compiled_image_with_abi, pred_pt, abi_pt, (0, 255, 255), 1)
          
      # Save the image with keypoints and lines
      cv2.imwrite(f'predicted_kp/RANSAC_{pair[0]}_{pair[1]}.png' if self.ransac_on else f'predicted_kp/{pair[0]}_{pair[1]}.png', compiled_image_with_abi)
           
      #print(mean_error)
      #print(len(errors))
      # Plot the histogram of error values
      plt.hist(errors, edgecolor='black')
      plt.xlabel('Error Value (Euclidean Distance)')
      plt.ylabel('Frequency')
      
      # Wrap the title
      title = f'{pair} WITH RANSAC' if self.ransac_on else f'{pair} WITHOUT RANSAC'
      wrapped_title = "\n".join(textwrap.wrap(title, width=60))
      plt.title(wrapped_title)
      
      plt.tight_layout()
      plt.savefig(f"test/RANSAC_{pair}.png" if self.ransac_on else f"test/NO_RANSAC_{pair}.png")
      plt.close()
      
  
test('images', False)