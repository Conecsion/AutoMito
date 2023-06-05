#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision
from PIL import Image


#  img_path = "/home/shaodi/Work/Embryo/yolov8/scripts/Embryo0689_crop.png"
#
#  masks = torch.load('/home/shaodi/Work/Embryo/yolov8/scripts/Embryo0689_crop_mask.pt')
#
#  def show_mask(mask, ax, random_color=False):
    #  if random_color:
        #  color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    #  else:
        #  color = np.array([30/255, 144/255, 255/255, 0.6])
    #  h, w = mask.shape[-2:]
    #  mask_image = mask.reshape(h,w,1) * color.reshape(1,1,-1)
    #  ax.imshow(mask_image)
#
#  def show_box(box, ax):
    #  x0, y0 = box[0], bbox[1]
    #  w, h = box[2] - box[0], box[3] - box[1]
    #  ax.add_patch(plt.Rectangle((x0, y0),w,h,edgecolor='green',facecolor=(0,0,0,0), lw=2))
#
#
#  image = cv2.imread(img_path)
#  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


#  plt.figure(figsize=(10,10))
#  plt.imshow(image)
#  for i in range(len(masks)):
    #  show_mask(masks[i], plt.gca())
#  plt.axis('off')
#  plt.show()

def get_mask_img(mask):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    color = np.array([30/255, 144/255, 255/255])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h,w,1) * color.reshape(1,1,-1)
    return mask_image


mask_dir = '/home/shaodi/Work/Embryo/yolov8/output/mask2'
#  mask_arr = []
for i in range(328,1441):
    if i < 1000:
        i = f"0{i}"
    else:
        i = f"{i}"
    maskfile = os.path.join(mask_dir, f"Embryo{i}_crop_mask.pt")
    if os.path.exists(maskfile):
        mask = torch.load(maskfile)
        mask = torch.sum(mask, dim=0).squeeze()
        mask = torch.clamp(mask, max=1)
        mask = mask.to(torch.uint8)
        #  mask = get_mask_img(mask)
    else:
        mask = torch.zeros((512,512))
        mask = mask.to(torch.uint8)
        #  mask = get_mask_img(torch.zeros((512,512)))
    mask = mask.numpy()
    mask = mask * 255
    print(np.where(mask > 0))
    im = Image.fromarray((mask).astype(np.uint8))
    im.save(os.path.join(mask_dir,'mask_images',f"{i}.png"), format="PNG")
    #  torchvision.utils.save_image(mask, os.path.join(mask_dir, "mask_images", "{i}.png"))


        
