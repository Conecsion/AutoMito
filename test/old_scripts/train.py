#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ultralytics import YOLO

model = YOLO('yolov8m.pt')

model.train(data='data.yaml', \
            epochs=100, \
            imgsz=512, \
            device='0,1,2', \
            project='yolov8m', \
            pretrained=True, \
            batch=30, \
            )

