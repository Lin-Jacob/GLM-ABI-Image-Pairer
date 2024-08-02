import torch
import cv2
import numpy as np
import os
import image_utils
from SuperGluePretrainedNetwork.models.superglue import SuperGlue
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import read_image, make_matching_plot, plot_image_pair, plot_keypoints, plot_matches
from PIL import Image


class find_match:
    def __init__(self, image_file) -> None:
        self.image_dir = image_file
        self.keypoints = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.find_matches()

    def find_matches(self):
        config = {
            'superpoint': {
                'descriptor_dim': 256,
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': -1,
                'remove_borders': 4,
            },
            'superglue': {
                'descriptor_dim': 256,
                'weights': 'indoor',
                'keypoint_encoder': [32, 64, 128, 256],
                'GNN_layers': ['self', 'cross'] * 9,
                'sinkhorn_iterations': 100,
                'match_threshold': 0.2,
            }
        }
        matcher = Matching(config).eval().to(self.device)

        # Get list of ABI and GLM images
        abi_images = []
        glm_images = []

        for filename in os.listdir(self.image_dir):
            if "ABI" in filename:
                abi_images.append(os.path.join(self.image_dir, filename))
            elif "GLM" in filename:
                glm_images.append(os.path.join(self.image_dir, filename))

        original_width, original_height = (1571, 1571)
        sub_width, sub_height = (170, 170)

        # Calculate the number of positions you will iterate through
        num_x_positions = original_width // sub_width

        reassembled_image = Image.new('L', (original_width, original_height))

        # Step 2 and 3: Match ABI with GLM images and save matched pairs with lines
        for glm_img_path in glm_images:
            glm_img, glm_input, glm_scales = read_image(
                path=glm_img_path,
                device=None,
                resize=[-1],
                rotation=0,
                resize_float=False)

            best_match = None
            best_abi_kp = []
            best_glm_kp = []
            best_abi_img = None
            curr_best_num_matches = 0

            for abi_img_path in abi_images:

                image_subsets = image_utils.split_image(
                    abi_img_path, (170, 170))
                for index, abi_subset in enumerate(image_subsets):
                    top = (index // num_x_positions) * sub_height
                    left = (index % num_x_positions) * sub_width

                    abi_subset.save('temp_abi_subset.png')
                    abi_img, abi_input, abi_scales = read_image(
                        path='temp_abi_subset.png',
                        device=self.device,
                        resize=[-1],
                        rotation=0,
                        resize_float=False)

                    glm_input = glm_input.to(self.device)
                    abi_input = abi_input.to(self.device)

                    predict = matcher(
                        {'image0': glm_input, 'image1': abi_input})
                    predict = {k: v[0].detach().cpu().numpy()
                               for k, v in predict.items()}
                    glm_keypoints, abi_keypoints = predict['keypoints0'], predict['keypoints1']
                    matches, confidence = predict['matches0'], predict['matching_scores0']

                    valid = matches > -1
                    matching_glm_kp = glm_keypoints[valid]
                    matching_abi_kp = abi_keypoints[matches[valid]]
                    matching_confidence = confidence[valid]

                    if len(matching_glm_kp) > curr_best_num_matches:
                        curr_best_num_matches = len(matching_glm_kp)
                        best_match = abi_img_path[7:]
                        best_glm_kp = matching_glm_kp
                        best_abi_kp = matching_abi_kp
                        best_abi_img = abi_img

            self.keypoints[(glm_img_path[7:], abi_img_path[7:])] = {
                'abi_keypoints': best_abi_kp,
                'glm_keypoints': best_glm_kp,
                'matched_keypoints': [(tuple(best_abi_kp[i]), tuple(best_glm_kp[i])) for i in range(len(best_abi_kp))]
            }


find_match('images')
