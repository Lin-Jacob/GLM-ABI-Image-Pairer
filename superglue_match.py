import cv2
import numpy as np
import os
import torch
from torch import nn
from SuperGluePretrainedNetwork.models.superglue import SuperGlue
from SuperGluePretrainedNetwork.models.utils import frame2tensor, make_matching_plot_fast


class find_match:
    def __init__(self, image_file) -> None:
        self.image_dir = image_file
        self.keypoints = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.superglue = SuperGlue({
            'descriptor_dim': 256,
            'weights': 'indoor',
            'keypoint_encoder': [32, 64, 128, 256],
            'GNN_layers': ['self', 'cross'] * 9,
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
        }).to(self.device)
        self.find_matches()

    def find_matches(self):
        abi_images = []
        glm_images = []

        for filename in os.listdir(self.image_dir):
            if "ABI" in filename:
                abi_images.append(os.path.join(self.image_dir, filename))
            elif "GLM" in filename:
                glm_images.append(os.path.join(self.image_dir, filename))

        sift = cv2.SIFT_create()

        for glm_img_path in glm_images:
            glm_img = cv2.imread(glm_img_path, cv2.IMREAD_GRAYSCALE)
            glm_keypoints, glm_descriptors = sift.detectAndCompute(
                glm_img, None)

            best_match = None
            best_match_distance = float('inf')

            for abi_img_path in abi_images:
                abi_img = cv2.imread(abi_img_path, cv2.IMREAD_GRAYSCALE)
                abi_keypoints, abi_descriptors = sift.detectAndCompute(
                    abi_img, None)

                # Convert keypoints and descriptors to tensors
                glm_keypoints_t = torch.tensor(
                    [kp.pt for kp in glm_keypoints], device=self.device).unsqueeze(0)
                abi_keypoints_t = torch.tensor(
                    [kp.pt for kp in abi_keypoints], device=self.device).unsqueeze(0)
                glm_descriptors_t = torch.tensor(
                    glm_descriptors, device=self.device).unsqueeze(0).transpose(1, 2)
                abi_descriptors_t = torch.tensor(
                    abi_descriptors, device=self.device).unsqueeze(0).transpose(1, 2)
                glm_scores_t = torch.ones(
                    (1, glm_keypoints_t.shape[1]), device=self.device)
                abi_scores_t = torch.ones(
                    (1, abi_keypoints_t.shape[1]), device=self.device)
                a = f"""glm_keypoints_t.shape: {glm_keypoints_t.shape},
                    abi_keypoints_t.shape: {abi_keypoints_t.shape},
                    glm_descriptors_t.shape: {glm_descriptors_t.shape},
                    abi_descriptors_t.shape: {abi_descriptors_t.shape},
                    glm_scores_t.shape: {glm_scores_t.shape},
                    abi_scores_t.shape: {abi_scores_t.shape}"""
                print(a)
                input_dict = {
                    'keypoints0': glm_keypoints_t,
                    'keypoints1': abi_keypoints_t,
                    'descriptors0': glm_descriptors_t,
                    'descriptors1': abi_descriptors_t,
                    'scores0': glm_scores_t,
                    'scores1': abi_scores_t,
                    'image0': frame2tensor(glm_img, self.device),
                    'image1': frame2tensor(abi_img, self.device)
                }

                with torch.no_grad():
                    pred = self.superglue(input_dict)

                matches = pred['matches0'][0].cpu().numpy()
                valid = matches > -1
                matched_kp1 = glm_keypoints_t[0, valid].cpu().numpy()
                matched_kp2 = abi_keypoints_t[0, matches[valid]].cpu().numpy()

                if len(matched_kp1) > 0:
                    distances = np.linalg.norm(
                        matched_kp1 - matched_kp2, axis=1)
                    mean_distance = np.mean(distances)
                    if mean_distance < best_match_distance:
                        best_match_distance = mean_distance
                        best_match = abi_img_path
                        good_matches = [(i, matches[i]) for i in range(
                            len(matches)) if matches[i] > -1]

            if best_match is not None:
                matched_folder = "test"
                os.makedirs(matched_folder, exist_ok=True)

                matched_abi_img = cv2.imread(best_match)
                matched_img = make_matching_plot_fast(
                    cv2.imread(glm_img_path), cv2.imread(best_match),
                    glm_keypoints_t[0].cpu().numpy(
                    ), abi_keypoints_t[0].cpu().numpy(), good_matches
                )

                glm_filename = os.path.basename(glm_img_path)
                abi_filename = os.path.basename(best_match)
                output_filename = f"SUPERGLUE_{glm_filename}_matched_with_{abi_filename}.jpg"
                output_path = os.path.join(matched_folder, output_filename)

                cv2.imwrite(output_path, matched_img)

                im1 = cv2.imread(os.path.join(self.image_dir, glm_filename))
                im2 = cv2.imread(os.path.join(self.image_dir, abi_filename))

                # Create KeyPoint objects for drawing
                glm_points = [glm_keypoints[m[0]].pt for m in good_matches]
                abi_points = [abi_keypoints[m[1]].pt for m in good_matches]

                glm_keypoints_cv = [cv2.KeyPoint(
                    x=p[0], y=p[1], size=1) for p in glm_points]
                abi_keypoints_cv = [cv2.KeyPoint(
                    x=p[0], y=p[1], size=1) for p in abi_points]

                dim1 = cv2.drawKeypoints(
                    im1, glm_keypoints_cv, None, color=(255, 255, 0))
                dim2 = cv2.drawKeypoints(
                    im2, abi_keypoints_cv, None, color=(255, 255, 0))

                cv2.imwrite(f"test/{glm_filename}", dim1)
                cv2.imwrite(f"test/{abi_filename}", dim2)

                print(len(good_matches), glm_filename, abi_filename)

                self.keypoints[(glm_filename, abi_filename)] = {
                    'abi_keypoints': [kp.pt for kp in abi_keypoints],
                    'glm_keypoints': [kp.pt for kp in glm_keypoints],
                    'matched_keypoints': [(abi_keypoints[m[1]].pt, glm_keypoints[m[0]].pt) for m in good_matches]
                }


# Example usage
find_match('images')
