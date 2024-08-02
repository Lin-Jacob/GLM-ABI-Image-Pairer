import cv2
import numpy as np
import os

class find_match:
    def __init__(self, image_file) -> None:
        self.image_dir = image_file
        self.keypoints = {}
        self.find_matches()
    
    def find_matches(self):
        # Step 1: Get list of ABI and GLM images
        abi_images = []
        glm_images = []

        for filename in os.listdir(self.image_dir):
            if "ABI" in filename:
                abi_images.append(os.path.join(self.image_dir, filename))
            elif "GLM" in filename:
                glm_images.append(os.path.join(self.image_dir, filename))
                
        sift = cv2.SIFT_create()
        
        # Step 2 and 3: Match ABI with GLM images and save matched pairs with lines
        for glm_img_path in glm_images:
            glm_img = cv2.imread(glm_img_path, cv2.IMREAD_GRAYSCALE)
            glm_keypoints, glm_descriptors = sift.detectAndCompute(glm_img, None)

            best_match = None
            best_match_distance = float('inf')

            for abi_img_path in abi_images:
                abi_img = cv2.imread(abi_img_path, cv2.IMREAD_GRAYSCALE)
                abi_keypoints, abi_descriptors = sift.detectAndCompute(abi_img, None)

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
                matched_img = cv2.drawMatches(matched_abi_img, abi_keypoints, glm_img, glm_keypoints, good_matches, None)

                glm_filename = os.path.basename(glm_img_path)
                abi_filename = os.path.basename(best_match)
                output_filename = f"{glm_filename}_matched_with_{abi_filename}.jpg"
                output_path = os.path.join(matched_folder, output_filename)

                cv2.imwrite(output_path, matched_img)

                im1 = cv2.imread(os.path.join(self.image_dir, glm_filename))
                im2 = cv2.imread(os.path.join(self.image_dir, abi_filename))

                
                # Create KeyPoint objects for drawing
                glm_points = [glm_keypoints[m.trainIdx].pt for m in good_matches]
                abi_points = [abi_keypoints[m.queryIdx].pt for m in good_matches]
                

                glm_keypoints_cv = [cv2.KeyPoint(x=p[0], y=p[1], size=1) for p in glm_points]
                abi_keypoints_cv = [cv2.KeyPoint(x=p[0], y=p[1], size=1) for p in abi_points]

                dim1 = cv2.drawKeypoints(im1, glm_keypoints_cv, None, color = (255,255,0))
                dim2 = cv2.drawKeypoints(im2, abi_keypoints_cv, None, color = (255,255,0))

                cv2.imwrite(f"test/{glm_filename}", dim1)
                cv2.imwrite(f"test/{abi_filename}", dim2)

                print(len(good_matches), glm_filename, abi_filename)
                
                self.keypoints[(glm_filename, abi_filename)] = {
                    'abi_keypoints': [m.pt for m in abi_keypoints],
                    'glm_keypoints': [m.pt for m in glm_keypoints],
                    'matched_keypoints': [(abi_keypoints[m.queryIdx].pt, glm_keypoints[m.trainIdx].pt) for m in good_matches]
                }

