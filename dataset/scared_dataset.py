import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from dataset.data_io import get_transform, read_all_lines
import tifffile

class ScaredDataset(Dataset):
    """
    SCARED 数据集加载类，用于读取立体图像对和对应的视差图
    支持训练/测试模式，可配置裁剪尺寸
    """

    def __init__(self, datapath, list_filename, training, crop_size=None):
        """
        参数:
            datapath: 数据集根目录
            list_filename: 包含左图、右图、视差图路径的文本文件
            training: True 为训练模式（随机裁剪），False 为测试模式（固定裁剪）
            crop_size: (w, h) 裁剪尺寸，若为None则使用默认值（训练:640x480，测试:1280x1024）
        """
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        self.img_width = 1280
        self.img_height = 1024

        # 设置裁剪尺寸
        if crop_size is not None:
            self.crop_w, self.crop_h = crop_size
        else:
            if training:
                self.crop_w, self.crop_h = 640, 480
            else:
                self.crop_w, self.crop_h = 1280, 1024

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_image(self, filename):
        img_path = os.path.normpath(os.path.join(self.datapath, filename))
        return Image.open(img_path).convert('RGB')

    def load_disp(self, filename):
        disp_path = os.path.join(self.datapath, filename)
        disp = tifffile.imread(disp_path)
        disp = np.array(disp, dtype=np.float32)
        disp = np.ascontiguousarray(disp)
        return disp

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(self.left_filenames[index])
        right_img = self.load_image(self.right_filenames[index])
        disparity = self.load_disp(self.disp_filenames[index])

        if self.training:
            # 训练模式：随机裁剪
            x1 = random.randint(0, self.img_width - self.crop_w)
            y1 = random.randint(0, self.img_height - self.crop_h)
            left_img = left_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
            right_img = right_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
            disparity = disparity[y1:y1 + self.crop_h, x1:x1 + self.crop_w]

            processed = get_transform()
            left_tensor = processed(left_img)
            right_tensor = processed(right_img)
            disparity = torch.from_numpy(disparity.copy()).float()

            return {
                "left": left_tensor,
                "right": right_tensor,
                "disparity": disparity
            }
        else:
            # 测试模式：从右下角固定裁剪（保证一致性）
            x1 = self.img_width - self.crop_w
            y1 = self.img_height - self.crop_h
            left_img = left_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
            right_img = right_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
            disparity = disparity[y1:y1 + self.crop_h, x1:x1 + self.crop_w]

            processed = get_transform()
            left_tensor = processed(left_img)
            right_tensor = processed(right_img)
            disparity = torch.from_numpy(disparity.copy()).float()

            return {
                "left": left_tensor,
                "right": right_tensor,
                "disparity": disparity,
                "top_pad": 0,
                "right_pad": 0,
                "left_filename": self.left_filenames[index]
            }