from crop import crop, crop_one
from yolo_mito_detect import yolo_mito_detect
from downsample import downsample
from format_convert import format_convert
import os

from torch.utils.data import DataLoader
import torch


class GeneratorDataset(torch.utils.data.IterableDataset):

    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return iter(self.generator)


if __name__ == "__main__":
    crop(512)
    results = yolo_mito_detect('0', 512, "model/yolo_mito.pt")
    #  dataset = GeneratorDataset(results)
    #  dataloader = DataLoader(dataset, batch_size=32)
    #  for batch_idx, result_batch in enumerate(dataloader):
    #  print(batch_idx)
    #  print(result_batch)
    for result in results:
        print(result.boxes.xyxy)
