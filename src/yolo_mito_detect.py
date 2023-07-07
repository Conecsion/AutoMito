#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import torch
import time
import csv
import sys
import numpy as np
import pandas as pd
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Tuple

sys.path.append("..")


def yolo_mito_detect(
    yolo_model='model/yolo_mito.pt',  # Path to yolo model
    input_path='output/cropped_images',
    csv_file='',
    output_path="output/YOLO_prediction",
    gpu_ids='0,1,2,3',  # GPU device to run on
    batch_size=1,  # Batch size per GPU
    img_size=(
        2048, 2048
    ),  # Input image size. Detection size should better match the image size when training model
    max_det=2000,  # Maximum number of detections per patch
    conf=0.4,  # Object confidence threshold for NMS
    iou=0.7,
):
    os.makedirs(output_path, exist_ok=True)
    print("---------------YOLO OBJECT DETECTING---------------")
    time.sleep(2)

    # Initialize queue
    queue = mp.Manager().Queue()

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    gpu_ids = gpu_ids.split(',')
    world_size = len(gpu_ids)

    mp.spawn(
        ddp,
        args=(
            world_size,
            input_path,
            csv_file,
            yolo_model,
            gpu_ids,
            batch_size,
            img_size,
            max_det,
            conf,
            iou,
            queue,
        ),
        nprocs=world_size,
        join=True,
    )
    queue.put("DONE")

    write_results(output_path, queue)

    print("---------------YOLO DETECTION DONE---------------")
    time.sleep(10)


class Yolo_det_dataset(Dataset):

    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        img = cv2.imread(os.path.join(self.image_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        if self.transform:
            img = self.transform(img)
        # return img, img_name
        return dict(img=img, img_name=img_name)


def ddp(rank, world_size, input_path, csv_file, yolo_model, gpu_ids,
        batch_size, img_size, max_det, conf, iou, queue):
    model = YOLO(yolo_model)
    print(gpu_ids[rank])
    dataset = Yolo_det_dataset(input_path, csv_file)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=64,
                            pin_memory=True,
                            sampler=dist_sampler)

    for batched_input in dataloader:
        batched_output = model(
            batched_input['img'],
            # stream=True,
            imgsz=img_size,
            max_det=max_det,
            conf=conf,
            iou=iou,
            device=gpu_ids[rank],
        )
        # for result in batched_output:
        for i in range(len(batched_output)):
            result = batched_output[i]
            boxes = result.boxes
            img_name = batched_input['img_name'][i]
            queue.put((img_name, boxes))


#     cleanup()

# def cleanup():
#     dist.destroy_process_group()


def write_results(output_path, queue):
    with open(os.path.join(output_path, 'yolo_detection_index.csv'), 'a') as f:
        writer = csv.writer(f)
        while True:
            index_line = queue.get()
            if index_line == "DONE":
                break
            else:
                writer.writerow(index_line)
