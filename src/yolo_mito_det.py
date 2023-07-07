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
from torch.utils.data import Dataset, DataLoader, sampler
import torch.multiprocessing as mp
from typing import Tuple

sys.path.append("..")


def yolo_mito_detect(
    yolo_model='model/yolo_mito.pt',  # Path to yolo model
    input_path='output/cropped_images',
    csv_file='',
    output_path="output/YOLO_prediction",
    gpu_ids='0,1,2,3',  # GPU device to run on
    img_size=(
        2048, 2048
    ),  # Input image size. Detection size should better match the image size when training model
    batch_size=16,  # Batch size per GPU
    max_det=2000,  # Maximum number of detections per patch
    conf=0.4,  # Object confidence threshold for NMS
    iou=0.7,
):
    os.makedirs(output_path, exist_ok=True)
    print("---------------YOLO OBJECT DETECTING---------------")
    time.sleep(2)

    # Initialize queue
    queue = mp.Manager().Queue()

    model = YOLO(yolo_model)
    dataset = Yolo_det_dataset(input_path, csv_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=64,
        pin_memory=True,
    )

    for batched_input in dataloader:
        # print(type(batched_input))
        # print(batched_input)
        batched_output = model(batched_input[0],
                               stream=True,
                               imgsz=img_size,
                               max_det=max_det,
                               conf=conf,
                               iou=iou,
                               device=0 / 1 / 2 / 3)
        for result in batched_output:
            boxes = result.boxes
            print(result)
            # print(boxes)
            # img_name = os.path.basename(result.path)
            img_name = ''
            queue.put((img_name, boxes))
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
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        if self.transform:
            img = self.transform(img)
        return img, self.dataframe.iloc[idx, 0]


def write_results(output_path, queue):
    with open(os.path.join(output_path, 'yolo_detection_index.csv'), 'a') as f:
        writer = csv.writer(f)
        while True:
            index_line = queue.get()
            if index_line == "DONE":
                break
            else:
                writer.writerow(index_line)
