import cv2
import numpy as np
import sift_match
import matplotlib.pyplot as plt
import textwrap

class homography():
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
  
      # Find the homography matrix
      H, _ = cv2.findHomography(np.array(glm_matched), np.array(abi_matched), cv2.RANSAC, 5) if self.ransac_on else cv2.findHomography(np.array(glm_matched), np.array(abi_matched))
  
      #print("Homography Matrix:\n", H)
  
      errors = []
  
      for i, glm_kp in enumerate(glm_matched):
          # Convert the GLM keypoint to homogeneous coordinates
          glm_point_homogeneous = np.append(glm_kp, 1)
  
          # Apply the homography transformation
          predicted_point_homogeneous = np.matmul(H, glm_point_homogeneous)
  
          # Convert back to Cartesian coordinates
          predicted_point = predicted_point_homogeneous[:2] / predicted_point_homogeneous[2]
  
          # Calculate the error (Euclidean distance)
          error = np.linalg.norm(predicted_point - abi_matched[i])
          errors.append(error)
  
          #print("Predicted point:", predicted_point)
          #print("GLM keypoint:", glm_kp)
          #print("ABI matched point:", abi_matched[i])
          #print("Matched keypoints:", sift_matched.keypoints[pair].get("matched_keypoints")[i])
          #print("Error:", error)

      mean_error = np.mean(errors)/len(errors)
      #print(mean_error)
      
      # Plot the histogram of error values
      plt.hist(errors, bins=20, edgecolor='black')
      plt.xlabel('Error Value (Euclidean Distance)')
      plt.ylabel('Frequency')
      
      # Wrap the title
      title = f'{pair} WITH RANSAC' if self.ransac_on else f'{pair} WITHOUT RANSAC'
      wrapped_title = "\n".join(textwrap.wrap(title, width=60))
      plt.title(wrapped_title)
      
      plt.tight_layout()
      plt.savefig(f"assessment_histograms/RANSAC_{pair}.png" if self.ransac_on else f"assessment_histograms/NO_RANSAC_{pair}.png")
      plt.close()
      
      self.get_assessment[pair] = {
        "H" : H,
        "mean_error" : mean_error
      }
  
#assessment('images', False)