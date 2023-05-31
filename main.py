from crop import crop
from yolo_mito_detect import yolo_mito_detect
from downsample import downsample
from format_convert import format_convert
import os

from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms
import torch

import matplotlib.pyplot as plt
import itertools

size = 512


class GeneratorDataset(IterableDataset):

    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator


if __name__ == "__main__":
    crop(size)
    results = yolo_mito_detect('0', size, "model/yolo_mito.pt")
    transform = transforms.ToTensor()
    # Plot image
    #  tensor = tensor.permute(1,2,0)
    #  plt.imshow(tensor)
    #  plt.show()

my_dict = [
    dict(
        zip(("image", "boxes", "original_size"),
            (transform(result.orig_img), result.boxes.xyxy,
             transform(result.orig_img).shape[:3]))) for result in results
]
print(my_dict)
