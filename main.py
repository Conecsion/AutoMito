from crop import crop
from yolo_mito_detect import yolo_mito_detect
from downsample import downsample
from format_convert import format_convert

import os
import sys

from torch.utils.data import IterableDataset, DataLoader
import torchvision.transforms as transforms
import torch

import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "model/sam_vit_h_4b8939.pth"
model_type = "vit_h"

size = 512
device = "cuda:0"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


class GeneratorDataset(IterableDataset):

    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator


def plot_tensor(tensor):
    tensor = tensor.permute(1,2,0)
    plt.imshow(tensor)
    plt.show()


if __name__ == "__main__":
    crop(size)
    results = yolo_mito_detect('0', size, "model/yolo_mito.pt")

    batched_input = [
        dict(
            zip(("image", "boxes", "original_size"),
                (result[0].to("cuda"), result[1],
                 result[0].shape[1:3]))) for result in results
    ]
    

    #  batched_output = sam(batched_input, multimask_output=False)

    BATCH_SIZE = 3
    inf_dataloader = DataLoader(GeneratorDataset(results), batch_size=BATCH_SIZE, num_workers=64)
    for batch in inf_dataloader:
        print(batch[0])
