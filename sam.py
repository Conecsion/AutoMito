#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as T
import os
import sys
from torch.utils.data import DataLoader
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

sam_checkpoint = "model/sam_vit_h_4b8939.pth"
model_type = 'vit_h'
img_dir = 'cropped'
# label_dir = '/home/shaodi/Work/Embryo/yolov8/runs/detect/predict9/labels'
output_dir = '/home/shaodi/Work/Embryo/yolov8/output/mask2'

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam = nn.DataParallel(sam, device_ids=[0, 1, 2, 3])
predictor = SamPredictor(sam.module)

for img_file in glob.glob(os.path.join(img_dir, "*.png")):
    img_name = os.path.relpath(img_file, img_dir).removesuffix('.png')
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img)

    label_file = os.path.join(label_dir, f"{img_name}.txt")
    #  if os.path.getsize(label_file) == 0:

    label = np.loadtxt(os.path.join(label_dir, f"{img_name}.txt"),
                       usecols=(1, 2, 3, 4))
    # Tell if there is any label box in this image
    if np.size(label) == 0:
        continue
    if label.ndim == 1:
        label = np.expand_dims(label, axis=0)

    columns = 512 * label
    coords = np.empty_like(columns)
    coords[:, 0:2] = columns[:, 0:2] - 0.5 * columns[:, 2:4]
    coords[:, 2:4] = columns[:, 0:2] + 0.5 * columns[:, 2:4]
    input_boxes = torch.from_numpy(coords)
    input_boxes = input_boxes.to(predictor.device)

    transformed_boxes = predictor.transform.apply_boxes_torch(
        input_boxes, img.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    torch.save(masks, os.path.join(output_dir, f"{img_name}_mask.pt"))

    #  for i, mask in enumerate(masks):
    #  mask = mask.numpy().astype(np.uint8)
    #  print(mask)
    #  cv2.imwrite(os.path.join(output_dir, f"{mask_file}_{i}_mask.png"), mask)


def sam_seg(yolo_results,
            device="cuda",
            input_img='output/cropped_images_png/',
            model_type='vit_h',
            sam_checkpoint="model/sam_vit_h_4b8939.pth"):
    data_loader = DataLoader(yolo_results, batch_size=32)
    device = torch.device("cuda")
    predictor = torch.nn.DataParallel(predictor).to(device)
    for batch_idx, result_batch in enumerate(data_loader):
        images = T.ToTensor()(result_batch.orig_img).to(device)
        prompts = result_batch
