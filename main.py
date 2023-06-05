#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from src.crop import crop
from src.yolo_mito_detect import yolo_mito_detect
from src.sam_mito_segmentation import sam_mito_segmentation
import argparse

GPU_COUNTS = torch.cuda.device_count()
default_gpu_ids = ",".join(str(i) for i in range(GPU_COUNTS))

parser = argparse.ArgumentParser(
    prog="YoSAM",
    description="Cooperate YOLOv8 and SAM to segment target object from images"
)
parser.add_argument('--input',
                    default='input',
                    help="Path to the directory of input images")
parser.add_argument('--output',
                    default='output',
                    help='Path to the directory of output masks')
parser.add_argument(
    '--gpu_ids',
    default=default_gpu_ids,
    help="The GPU used to detect and segment. Default to use all available GPUs"
)
parser.add_argument("--crop_size", type=int, default=512)
parser.add_argument("--yolo_model", default="model/yolo_mito.pt")
parser.add_argument("--sam_model", default="model/sam_vit_h_4b8939.pth")
parser.add_argument("--sam_type", default="vit_h")
parser.add_argument("--sam_batch_size", type=int, default=3)

args = parser.parse_args()

# These directories will be created if not present
CROP_OUTPUT = 'tmp/crop'
DOWNSAMPLE_OUTPUT = 'tmp/downsample'
YOLO_OUTPUT = "tmp/yolo"

if __name__ == "__main__":
    # Crop all input images into CROP_SIZE smaller images
    crop(args.crop_size, args.input, CROP_OUTPUT)

    # Use YOLOv8 model to generate detection boxes for target. Here the model is trained to detect mitochondria.
    yolo_mito_detect(args.gpu_ids, args.crop_size, args.yolo_model,
                     CROP_OUTPUT, YOLO_OUTPUT)

    # Use Segment Anything Model (SAM) to generate from the YOLOv8 detection boxes
    sam_mito_segmentation()
