import torch
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
from PIL import Image
import random
import numpy as np
import cv2  # 记得导入 cv2

class Degradation(object):
    def __init__(self, args):
        super(Degradation, self).__init__()
        self.args = args
        self.toTensor = ToTensor()
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

    # --- 核心修改 1: 去掉下划线，只返回噪声图 ---
    def add_gaussian_noise(self, clean_patch, sigma):
        """
        通用高斯噪声函数
        clean_patch: numpy array [H, W, C]
        sigma: 噪声强度 (如 15, 40, 50)
        """
        # 确保输入是 float 以便计算
        if clean_patch.dtype != np.float32:
             clean_patch = clean_patch.astype(np.float32)
             
        noise = np.random.randn(*clean_patch.shape)
        # 计算噪声图
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        
        # Dataset 里只想要处理后的图，不需要再把 clean_patch 传回去
        return noisy_patch

    # --- 新增: 支持 deblur (如果你之后要用 deblur_XX) ---
    def add_blur(self, img, kernel_size):
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # --- 兼容旧代码逻辑 (保留给旧项目用) ---
    def _degrade_by_type(self, clean_patch, degrade_type):
        # 这是一个兼容层，把旧的 0/1/2 映射到新的通用函数上
        if degrade_type == 0:
            sigma = 15
        elif degrade_type == 1:
            sigma = 25
        elif degrade_type == 2:
            sigma = 50
        else:
            sigma = 15 # 默认

        noisy = self.add_gaussian_noise(clean_patch, sigma)
        return noisy, clean_patch

    def single_degrade(self, clean_patch, degrade_type=None):
        if degrade_type is None:
            degrade_type = random.randint(0, 3)
        
        # 调用兼容层
        degrad_patch, _ = self._degrade_by_type(clean_patch, degrade_type)
        return degrad_patch