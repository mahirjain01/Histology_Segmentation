import glob
import os

import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import Grayscale, RandomCrop
from src.data import DatasetType, register_dataset


@register_dataset(DatasetType.UNLABALLED_DATASET)
class Histology(Dataset):
    def __init__(self, image_directory: str, epsilon: float = 0.05,crop_size: int = 250):
        image_files = glob.glob(os.path.join(image_directory, 'images', '*.png'))
        images = map(Image.open, image_files)
        images = map(ToTensor(), images)
        self.images = list(images)
        self.epsilon = epsilon
        self.cropper = RandomCrop(crop_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx][:3]  # Remove alpha band
        image = self.cropper(image)
        return (1 - self.epsilon) * image + self.epsilon


@register_dataset(DatasetType.LABELLED_DATASET)
class HistologyValidation(Dataset):
    def __init__(self, directory: str, epsilon: float = 0.05):
        image_files = glob.glob(os.path.join(directory, 'images', '*.png'))
        mask_files = [
            os.path.join(directory, "masks", os.path.basename(image_file))
            for image_file in image_files
        ]

        images = [Image.open(image_file) for image_file in image_files]
        images = [ToTensor()(image) for image in images]
        self.images = images

        masks = [Image.open(mask_file) for mask_file in mask_files]
        masks = [(ToTensor()(mask) > 0).float() for mask in masks]  # Convert mask to binary tensor
        self.masks = masks

        self.epsilon = epsilon

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx][:3]  # Remove alpha channel if present
        mask = self.masks[idx]
        return (1 - self.epsilon) * image + self.epsilon, mask.unsqueeze(0)
