import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils1.utils_image as util
import cv2


class Dataset(data.Dataset): 
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """
    def __init__(self, opt):
        super(Dataset, self).__init__()
        # print('Dataset: MEF for Multi-exposure Image Fusion.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['resize_size'] if opt['resize_size'] else 64
        self.crop_size = opt['crop_size'] if opt['crop_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_A = util.get_image_paths(opt['dataroot_A'])
        self.paths_B = util.get_image_paths(opt['dataroot_B'])

    def __getitem__(self, index):

        # ------------------------------------
        # get under-exposure image, over-exposure image
        # and norm-exposure image
        # ------------------------------------
        # print('input channels:', self.n_channels)
        A_path = self.paths_A[index]
        B_path = self.paths_B[index]
        img_A,_ = util.imread_uint(A_path, self.n_channels)
        img_B,_ = util.imread_uint(B_path, self.n_channels)

        if self.opt['phase'] == 'train': 
            """
            # --------------------------------
            # get under/over/norm patch pairs
            # --------------------------------
            """
            H, W, _ = img_A.shape
            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            #随机裁剪
            rnd_h = random.randint(0, max(0, H - self.crop_size))
            rnd_w = random.randint(0, max(0, W - self.crop_size))
            patch_A = img_A[rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size,:]
            patch_B = img_B[rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size,:]

            # 使用cv2.resize对裁剪后的patch进行resize
            patch_A = cv2.resize(patch_A, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
            patch_B = cv2.resize(patch_B, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0,7)
            # print('img_A shape:', img_A.shape)
            
            patch_A, patch_B = util.augment_img(patch_A, mode=mode), util.augment_img(patch_B, mode=mode)
            img_A = util.uint2tensor3(patch_A)
            img_B = util.uint2tensor3(patch_B)

            return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}

        else: 
            """
            # --------------------------------
            # get under/over/norm image pairs
            # --------------------------------
            """
            img_A = util.uint2single(img_A)
            img_B = util.uint2single(img_B)

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_A = util.single2tensor3(img_A)
            img_B = util.single2tensor3(img_B)

            return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}

    def __len__(self):
        return len(self.paths_A)
