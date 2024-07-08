import cv2
import numpy as np
import os

class sift_match:
    def __init__(self, image_file) -> None:
        self.image_dir = image_file
        self.keypoints = {}
        self.sift_match()
    
    def sift_match(self):
        # Step 1: Get list of ABI and GLM images
        abi_images = []
        glm_images = []

        for filename in os.listdir(self.image_dir):
            if "ABI" in filename:
                abi_images.append(os.path.join(self.image_dir, filename))
            elif "GLM" in filename:
                glm_images.append(os.path.join(self.image_dir, filename))

        # Step 2 and 3: Match ABI with GLM images and save matched pairs with lines
        for glm_img_path in glm_images:
            glm_img = cv2.imread(glm_img_path, cv2.IMREAD_GRAYSCALE)
            glm_keypoints, glm_descriptors = cv2.SIFT_create().detectAndCompute(glm_img, None)

            best_match = None
            best_match_distance = float('inf')

            for abi_img_path in abi_images:
                abi_img = cv2.imread(abi_img_path, cv2.IMREAD_GRAYSCALE)
                abi_keypoints, abi_descriptors = cv2.SIFT_create().detectAndCompute(abi_img, None)

                matcher = cv2.BFMatcher()
                matches = matcher.knnMatch(abi_descriptors, glm_descriptors, k=2)

                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                if len(good_matches) > 0:
                    mean_distance = np.mean([match.distance for match in good_matches])
                    if mean_distance < best_match_distance:
                        best_match_distance = mean_distance
                        best_match = abi_img_path

            if best_match is not None:
                matched_folder = "matched_output"
                os.makedirs(matched_folder, exist_ok=True)

                matched_abi_img = cv2.imread(best_match)
                matched_img = cv2.drawMatches(abi_img, abi_keypoints, glm_img, glm_keypoints, good_matches, None)

                glm_filename = os.path.basename(glm_img_path)
                abi_filename = os.path.basename(best_match)
                output_filename = f"{glm_filename}_matched_with_{abi_filename}.jpg"
                output_path = os.path.join(matched_folder, output_filename)

                cv2.imwrite(output_path, matched_img)
                
                self.keypoints[(glm_filename, abi_filename)] = {
                    'abi_keypoints': [m.pt for m in abi_keypoints],
                    'glm_keypoints': [m.pt for m in glm_keypoints],
                    'matched_keypoints': [(abi_keypoints[m.queryIdx].pt, glm_keypoints[m.trainIdx].pt) for m in good_matches]
                }

