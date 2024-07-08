import cv2
import numpy as np
import os

# Step 1: Get list of ABI and GLM images
images_dir = "images"
abi_images = []
glm_images = []

for filename in os.listdir(images_dir):
    if "ABI" in filename:
        abi_images.append(os.path.join(images_dir, filename))
    elif "GLM" in filename:
        glm_images.append(os.path.join(images_dir, filename))

# Step 2 and 3: Match ABI with GLM images and save matched pairs with lines
for glm_img_path in glm_images:
    glm_img = cv2.imread(glm_img_path, cv2.IMREAD_GRAYSCALE)
    glm_keypoints, glm_descriptors = cv2.SIFT_create().detectAndCompute(glm_img, None)
    
    best_match = None
    best_match_distance = float('inf')
    best_match_points = 0
    best_abi_keypoints = 0

    for abi_img_path in abi_images:
        abi_img = cv2.imread(abi_img_path, cv2.IMREAD_GRAYSCALE)
        abi_keypoints, abi_descriptors = cv2.SIFT_create().detectAndCompute(abi_img, None)

        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(abi_descriptors, glm_descriptors, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

            mean_distance = np.mean([match.distance for match in good_matches])
            if mean_distance < best_match_distance:
                best_match_distance = mean_distance
                best_match = abi_img_path
                best_match_points = len(good_matches)
                best_abi_keypoints = len(abi_keypoints)
    
    matched_pairs = [(abi_keypoints[pair.queryIdx].pt, glm_keypoints[pair.trainIdx].pt) for pair in good_matches]
    
    for pair in matched_pairs:
        print(pair)
                
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

        print(f"Match found: {glm_filename} with {abi_filename}") 
        print(f"Number of keypoints in GLM image {os.path.basename(glm_img_path)}: {len(glm_keypoints)}")
        print(f"Number of keypoints in ABI image {abi_filename}: {best_abi_keypoints}")
        print(f"Number of matched keypoints: {best_match_points}")
        print(f"Accuracy (matched / total points found): {(best_match_points / (best_abi_keypoints + len(glm_keypoints)))}")
        print("----------------------------------------------------------------------------------------------------------")

