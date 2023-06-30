#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from src.crop import crop
from src.yolo_mito_detect import yolo_mito_detect
from src.sam_mito_segmentation import sam_mito_segmentation
from src.merge import merge, merge_one, generate_blank_masks
from src.downsample import downsample, upsample_masks
from src.project_creator import project_creator
import argparse

GPU_COUNTS = torch.cuda.device_count()
default_gpu_ids = ",".join(str(i) for i in range(GPU_COUNTS))

parser = argparse.ArgumentParser(
    prog="YoSAM",
    description="Cooperate YOLOv8 and SAM to segment target object from images"
)
# Directory of input images
parser.add_argument('--input',
                    default='input',
                    help="Path to the directory of input images")
# Directory of final merged images and masks
parser.add_argument('--path',
                    default='.',
                    help='Path to the directory of output results')
# Directory of intermediate files, including cropped images, YOLO prediction results, SAM segmentations, downsampled images and so on
parser.add_argument("--cache_dir", default="tmp")
# Which GPU(s) to use. e.g. --gpu_ids "0,1" use the first two GPUs
parser.add_argument(
    '--gpu_ids',
    default=default_gpu_ids,
    help="The GPU used to detect and segment. Default to use all available GPUs"
)

parser.add_argument("--crop_size", type=int, default=512)
parser.add_argument("--yolo_model", default="./model/mito_det_yolov8n.pt")
parser.add_argument("--sam_checkpoint", default="model/sam_vit_h_4b8939.pth")
parser.add_argument("--sam_type", default="vit_h")
# The maximum batch size of SAM inference is 3 for each GPU, according to former tests
parser.add_argument("--sam_batch_size", type=int, default=3)

parser.add_argument('--yolo_output', default='tmp/yolo')
parser.add_argument('--exist_ok', action='store_true')

args = parser.parse_args()

# These directories will be created if not present
project_path = project_creator(args.path, exist_ok=args.exist_ok)
#  CROP_OUTPUT = os.path.join(project_path, "crop")
#  DOWNSAMPLE_OUTPUT = os.path.join(project_path, "downsample")
#  DETECTION_OUTPUT = os.path.join(project_path, 'detection')
#  SEGMENTATION_OUTPUT = os.path.join(project_path, "segmentation")
#  OUTPUT = os.path.join(project_path, 'output')

if __name__ == "__main__":
    # Crop all input images into CROP_SIZE smaller images
    #  crop(args.crop_size, args.input, CROP_OUTPUT)
    # crop(args.crop_size, args.input, os.path.join(project_path, 'crop'))

    # Use YOLOv8 model to generate detection boxes for target. Here the model is trained to detect mitochondria.
     yolo_mito_detect(args.gpu_ids, args.crop_size, args.yolo_model, 'input', 'whole_img_yolo')

    # Use Segment Anything Model (SAM) to generate from the YOLOv8 detection boxes
    #  sam_mito_segmentation(args.gpu_ids, args.sam_batch_size, args.yolo_output,
    #  SAM_OUTPUT, args.sam_checkpoint, args.sam_type)

    #  Merge cropped images and masks
    #  merge(args.crop_size, args.input, args.output, CROP_OUTPUT, args.yolo_output,
    #  SAM_OUTPUT)

    # Downsample original images for cell detection
    #  downsample(8, 'input', 'tmp/downsampled_img', 'tif')
    # Detect Cells with YOLO
    #  yolo_mito_detect(args.gpu_ids, (1258, 1436), 'model/cell_det_yolov8x.pt',
    #  'tmp/downsampled_img', 'tmp/cell_detection', 3, 16)

    # Segment Cells with SAM using YOLO detection boxes
    #  sam_mito_segmentation(args.gpu_ids, 3, 'tmp/cell_detection',
    #  'tmp/cell_segementation_downsampled')

    # Upsample cell masks
    #  upsample_masks('output/cell_segementation_2', 'input', 'output/cell_masks')
