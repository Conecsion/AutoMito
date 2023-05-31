import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transform
from torch.utils.data import DataLoader, IterableDataset
import cv2
import numpy as np
from PIL import Image
from segment_anything.utils.transforms import ResizeLongestSide


def prepare_image(image):
    print(image.shape)
    image = torch.as_tensor(image)
    print(image.shape)
    return image.permute(2, 0, 1).contiguous()


img = Image.open("output/cropped_images_png/Embryo1004_0_17.png")
arr1 = np.array(img)
print(arr1.shape)

arr2 = cv2.imread("output/cropped_images_png/Embryo1004_0_17.png")
print(arr2.shape)

arr3 = prepare_image(arr1)
print(arr3.shape)
