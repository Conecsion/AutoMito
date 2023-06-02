#!/usr/bin/env python
# -*- coding: utf-8 -*-

from multiprocessing import set_start_method
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
from segment_anything import sam_model_registry
import pandas as pd
import itertools
import torch.multiprocessing as mp
try:
    set_start_method('spawn')
except RuntimeError:
    pass
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

#  os.environ['MASTER_ADDR'] = 'localhost'
#  os.environ['RANK'] = '0'
#  os.environ['WORLD_SIZE'] = '4'
#  os.environ['MASTER_PORT'] = '12355'
#  os.environ['NCCL_SOCKET_IFNAME'] = 'eno1'


class SamDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __getitem__(self, idx):
        image = ToTensor()(Image.open(self.dataframe.iloc[idx, 0]))
        label = torch.load(self.dataframe.iloc[idx, 1])

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


def sam(rank,
        world_size=4,
        BATCH_SIZE=4,
        sam_checkpoint="model/sam_vit_h_4b8939.pth",
        model_type="vit_h"):

    # Regist SAM Model

    # Load Dataset and build up Dataloader
    dataset = SamDataset("output/YOLO_prediction/yolo_prediction.csv")
    dist_sampler = torch.utils.data.distributed.DistributedSamper(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            collate_fn=collate_fn,
                            sampler=dist_sampler)

    # Init DDP
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    # Registry SAM Model and migrate it to a single GPU
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    device = torch.device("cuda", rank)
    sam = sam.to(device)

    # Wrap SAM Model with DDP
    sam = DDP(sam, device_ids=[rank], output_device=rank)

    #  torch.distributed.init_process_group(backend="nccl")
    #  print(torch.distributed.is_initialized())
    #  print(torch.distributed.is_nccl_available())
    #  print(torch.distributed.is_torchelastic_launched())


def dist_run(sam_fn, world_size=4):
    mp.spawn(sam_fn, args=(world_size, ), nprocs=world_size, join=True)
