#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import pandas as pd

# Import DDP related packages
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os

#  from multiprocessing import set_start_method
#  import itertools
#  from torch.multiprocessing import Process, Pool
#  try:
#  set_start_method('spawn')
#  except RuntimeError:
#  pass
#  import os
#  import torch.distributed as dist


class SamDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __getitem__(self, idx):
        image = ToTensor()(Image.open(self.dataframe.iloc[idx, 0]))
        image.requires_grad = False
        label = torch.load(self.dataframe.iloc[idx, 1])
        label.requires_grad = False

        return dict(
            zip(("image", "boxes", "original_size"),
                (image, label, image.shape[1:3])))

    def __len__(self):
        return len(self.dataframe)


def collate_fn(data):
    batched_input = []
    for i in range(len(data)):
        batched_input.append(data[i])
    return batched_input


def sam_init(device="cuda",
             sam_checkpoint="model/sam_vit_h_4b8939.pth",
             model_type="vit_h"):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam


def ddp(rank, world_size, batch_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    model = sam_init(rank)
    ddp_model = DDP(model, device_ids=[rank])

    dataset = SamDataset("output/YOLO_prediction/yolo_prediction.csv")
    dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=dist_sampler,
                            collate_fn=collate_fn)

    for batched_input in dataloader:
        batched_output = ddp_model(batched_input, multimask_output=False)
        print(batched_output)

    cleanup()


def cleanup():
    dist.destroy_process_group()


def main(world_size=4, batch_size=3):
    mp.spawn(ddp, args=(world_size, batch_size), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
