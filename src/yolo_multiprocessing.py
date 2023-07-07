#!/usr/bin/env python

import torch
from ultralytics import YOLO

# model = YOLO('/data/shaodi/DeepAi/yolo/DeepAI_Mito_Det/230706/weights/best.pt')
# model = torch.load(
#     '/data/shaodi/DeepAi/yolo/DeepAI_Mito_Det/230706/weights/best.pt')

# model = torch.hub.load()
model = torch.hub.load(
    '/data/shaodi/DeepAi/yolo/DeepAI_Mito_Det/230706/weights/best.pt',
    'yolov8x',
    source='local',
    pretrained=True,
)
