#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from ultralytics import YOLO
import torch
import time


def yolo_mito_detect(device='0,1,2,3',
                     size=512,
                     yolo_model='model/yolo_mito.pt',
                     input='output/cropped_images/*.tif',
                     output="output/YOLO_prediction"):
    os.makedirs(output, exist_ok=True)
    print("---------------YOLO OBJECT DETECTING---------------")
    time.sleep(2)

    model = YOLO(yolo_model)
    results = model.predict(source=input, \
                conf=0.4, \
                line_width=1, \
                show_labels=False, \
                imgsz=size, \
                device=device, \
                max_det=2000, \
                stream=True)
    # Save Prediction results
    for result in results:
        # Generate a new generator every time call this function
        yield result

        # Save prediction results
        filename = os.path.splitext(os.path.basename(result.path))[0] + ".pt"
        filename = os.path.join(output, filename)
        #  print(result.boxes.xyxy)
        if os.path.isfile(filename):
            continue
        else:
            torch.save(result.boxes.xyxy, filename)

    print("---------------YOLO DETECTION DONE---------------")
