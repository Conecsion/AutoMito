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


def sam_init(BATCH_SIZE=4,
             sam_checkpoint="model/sam_vit_h_4b8939.pth",
             model_type="vit_h"):

    # Regist SAM Model

    # Registry SAM Model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    return sam


def sam_cal(model, batched_input):
    batched_output = model(batched_input, multimask_output=False)


def MultiprocessingSam(batch_size=4, gpu_ids=[0, 1, 2, 3]):

    # Load Dataset and build up Dataloader
    dataset = SamDataset("output/YOLO_prediction/yolo_prediction.csv")
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn)

    num_processes = len(gpu_ids)
    model = sam_init()
    model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=sam_cal, args=(
            model,
            batched_input,
        ))
        p.start
        processes.append(p)
    for p in processes:
        p.join()
