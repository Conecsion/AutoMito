#!/usr/bin/env python

import torch
from ultralytics import YOLO

# model = YOLO('/data/shaodi/DeepAi/yolo/DeepAI_Mito_Det/230706/weights/best.pt')
model = torch.load(
    '/data/shaodi/DeepAi/yolo/DeepAI_Mito_Det/230706/weights/best.pt')
# model.export(format="-")

print(model.state_dict())
