import glob
import os

import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import Grayscale, RandomCrop
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.histology import Histology, HistologyValidation

# Initialize the datasets
# histology_dataset = Histology(image_directory='../HistologyNet/unlabelled')
validation_dataset = HistologyValidation(directory='../HistologyNet/labelled')

# Create data loaders for the datasets
batch_size = 4
# histology_dataloader = DataLoader(histology_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Iterate over the data loader and visualize the images
# for batch_idx, images in enumerate(histology_dataloader):
#     for i, image in enumerate(images):
#         plt.subplot(1, len(images), i + 1)
#         plt.imshow(image.permute(1, 2, 0).numpy())
#         image_np = TF.to_pil_image(image)  
#         plt.imshow(image_np)
#         plt.axis('off')
#     plt.show()
#     if batch_idx >= 3:
#         break

# Iterate over the validation data loader and visualize the images and masks
for batch_idx, (images, masks) in enumerate(validation_dataloader):
    for i, (image, mask) in enumerate(zip(images, masks)):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image.transpose(0, 2).numpy())  # Transpose to (H, W, C)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mask.transpose(0, 2).numpy().squeeze(), cmap='gray')  
        plt.axis('off')

        plt.show()
        if batch_idx >= 3:
            break
    if batch_idx >= 3:
        break
