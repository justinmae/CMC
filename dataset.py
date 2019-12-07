from __future__ import print_function

import numpy as np
from skimage import color
import torch
import torchvision.datasets as datasets
from torchvision import transforms
import torchvision.transforms.functional as TF

class ImageFolderInstance(datasets.ImageFolder):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index)
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class DatasetInstance(torch.utils.data.Dataset):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, dataset, two_crop=False):
        super(DatasetInstance, self).__init__()
        self.two_crop = two_crop
        self.dataset = dataset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        img, target = self.dataset[index]

        if self.two_crop:
            img2, _ = self.dataset[index]
            img = torch.cat([img, img2], dim=0)

        return img, target, index

    def __len__(self):
        return len(self.dataset)

class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, img):

        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)
        return np.float32(img)

class Rotation(object):
    def __call__(self, img):
        # rot_img = transforms.Compose([transforms.RandomRotation(45)])(img)
        # rot_img = np.asarray(rot_img)
        # img = np.asarray(img)
        rot_img = TF.rotate(img, 45)
        img = np.concatenate((img,rot_img),2)
        return img

class LabRotMix(object):
    def __call__(self, img):
        rot_img = TF.rotate(img, 45)
        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)
        rot_img = np.asarray(rot_img, np.uint8)
        rot_img = color.rgb2lab(rot_img)
        return np.float32(np.concatenate((img[:,:,:1], rot_img[:,:,1:]),2))
