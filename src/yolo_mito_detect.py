#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from ultralytics import YOLO
import torch
import time
import torchvision.transforms as transforms
import csv
import sys

sys.path.append("..")


def yolo_mito_detect(
    device='0,1,2,3',
    size=(512, 512),
    yolo_model='model/yolo_mito.pt',
    input_dir='output/cropped_images',
    output_dir="output/YOLO_prediction",
    batch=16,
    max_det=2000,
    conf=0.4,
):
    os.makedirs(output_dir, exist_ok=True)
    print("---------------YOLO OBJECT DETECTING---------------")
    time.sleep(2)

    model = YOLO(yolo_model)
    results = model.predict(source=input_dir, \
                conf=conf, \
                line_width=1, \
                show_labels=False, \
                imgsz=size, \
                device=device, \
                max_det=max_det, \
                stream=True,
                save=True,
                batch=batch,
                exist_ok=True,
                save_txt=True,)

    with open(os.path.join(output_dir, "yolo_prediction.csv"), "w",
              newline="") as f:
        writer = csv.writer(f)

        # Save Prediction results
        for result in results:
            label = result.boxes.xyxy
            if label.numel() != 0:
                img_path = result.path
                root, _ = os.path.splitext(img_path)
                name = os.path.basename(root)
                label_path = os.path.join(output_dir, name + ".pt")
                if not os.path.isfile(label_path):
                    torch.save(label, label_path)
                writer.writerow([img_path, label_path])
            else:
                continue

    print("---------------YOLO DETECTION DONE---------------")
    time.sleep(10)
