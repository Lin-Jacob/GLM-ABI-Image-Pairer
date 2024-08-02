import cv2
import numpy as np
import sift_match
import flann_match
import matplotlib.pyplot as plt
import textwrap
import superpoint_superglue_match
from PIL import Image


class homography():
    def __init__(self, img1, img2, super_glue, ransac_on: bool):
        self.image1 = img1
        self.image2 = img2
        self.super_glue_results = super_glue
        self.ransac_on = ransac_on
        self.get_assessment = {}
        self.accuracy_assessment()

    def accuracy_assessment(self):
        matching_type = 'SUPERGLUE'
        superglue_matched = self.super_glue_results

        for (pair, keypoints) in superglue_matched.keypoints.items():
            abi_matched, glm_matched = zip(*keypoints.get("matched_keypoints"))

            # Find the homography matrix
            H, _ = cv2.findHomography(np.array(glm_matched), np.array(
                abi_matched), cv2.RANSAC, 5) if self.ransac_on else cv2.findHomography(np.array(glm_matched), np.array(abi_matched))

            # print("Homography Matrix:\n", H)

            errors = []
            predicted_points = []

            for i, glm_kp in enumerate(glm_matched):
                # Convert the GLM keypoint to homogeneous coordinates
                glm_point_homogeneous = np.array([glm_kp[0], glm_kp[1], 1.0])

                # Apply the homography transformation
                predicted_point_homogeneous = np.dot(H, glm_point_homogeneous)

                # Convert back to Cartesian coordinates
                # print(predicted_point_homogeneous[:2], predicted_point_homogeneous[2])
                predicted_point = predicted_point_homogeneous[:2] / \
                    predicted_point_homogeneous[2]

                error = np.linalg.norm(predicted_point - abi_matched[i])
                errors.append(error)

                predicted_points.append(tuple(x for x in predicted_point))

            # Read the image for displaying keypoints
            compiled_image = cv2.imread(f"images/{pair[1]}")

            # Convert points to cv2.KeyPoint objects with larger size
            predicted_keypoints = [cv2.KeyPoint(
                x=float(p[0]), y=float(p[1]), size=1) for p in predicted_points]
            abi_keypoints = [cv2.KeyPoint(
                x=float(p[0]), y=float(p[1]), size=1) for p in abi_matched]

            # Draw keypoints on the image
            compiled_image_with_predicted = cv2.drawKeypoints(
                compiled_image, predicted_keypoints, None, color=(0, 0, 255))
            compiled_image_with_abi = cv2.drawKeypoints(
                compiled_image_with_predicted, abi_keypoints, None, color=(255, 0, 0))

            # Draw lines between the matched and predicted points
            for pred_kp, abi_kp in zip(predicted_keypoints, abi_keypoints):
                pred_pt = (int(pred_kp.pt[0]), int(pred_kp.pt[1]))
                abi_pt = (int(abi_kp.pt[0]), int(abi_kp.pt[1]))
                cv2.line(compiled_image_with_abi, pred_pt,
                         abi_pt, (0, 255, 255), 1)

            # Save the image with keypoints and lines
            cv2.imwrite(
                f'predicted_kp/{matching_type}_RANSAC_{pair[0]}_{pair[1]}.png' if self.ransac_on else f'predicted_kp/{matching_type}_{pair[0]}_{pair[1]}.png', compiled_image_with_abi)

            mean_error = np.mean(errors)/len(errors)
            # print(mean_error)
            # print(len(errors))
            # Plot the histogram of error values
            plt.hist(errors, edgecolor='black')
            plt.xlabel('Error Value (Euclidean Distance)')
            plt.ylabel('Frequency')

            # Wrap the title
            title = f'{matching_type}_{pair} WITH RANSAC' if self.ransac_on else f'{matching_type}_{pair} WITHOUT RANSAC'
            wrapped_title = "\n".join(textwrap.wrap(title, width=60))
            plt.title(wrapped_title)

            plt.tight_layout()
            plt.savefig(
                f"assessment_histograms/{matching_type}_RANSAC_{pair}.png" if self.ransac_on else f"assessment_histograms/{matching_type}_NO_RANSAC_{pair}.png")
            plt.close()

            self.get_assessment[pair] = {
                "H": H,
                "mean_error": mean_error
            }


homography('images', True)
