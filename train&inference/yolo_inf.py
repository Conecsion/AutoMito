#!/usr/bin/env python

from os import defpath
from ultralytics import YOLO

model = YOLO('model/mito_det_yolov8n.pt')

results = model('Embryo1045.tif')

for result in results:
    boxes = result.boxes
