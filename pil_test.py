#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import torch
from copy import deepcopy
from typing import Tuple
import torchvision.transforms as transforms
from segment_anything import sam_model_registry
import matplotlib.pyplot as plt
import numpy as np

torch.cuda.set_device(1)
sam = sam_model_registry['vit_h'](checkpoint='model/sam_vit_h_4b8939.pth')
sam.to(device='cuda')

img = Image.open('./Embryo0418_bin8_downsampled.tif').convert('RGB')


def get_preprocess_shape(oldh: int, oldw: int,
                         long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


#  def resize_transform(image:torch.Tensor, target_length: int) -> torch.Tensor:
#  target_size = get_preprocess_shape(image.shape[1], image.shape[2], target_length)


# Notice PIL.Image.size is in (W, H) format
def prepare_pil_image(image: Image.Image, long_side_length: int,
                      device) -> torch.Tensor:
    target_size = get_preprocess_shape(image.size[1], image.size[0],
                                       long_side_length)
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(target_size),
    ])
    transformed_image = transform(image)
    transformed_image = transformed_image.to(device.device)
    return transformed_image


def apply_coords_torch(coords: torch.Tensor, original_size: Tuple[int, ...],
                       target_length: int) -> torch.Tensor:
    """
    Expects a torch tensor with length 2 in the last dimension. Requires the
    original image size in (H, W) format.
    """
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(original_size[0], original_size[1],
                                        target_length)
    coords = deepcopy(coords).to(torch.float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords


def prepare_boxes(boxes: torch.Tensor, original_size: Tuple[int, ...],
                  target_length: int):
    """
    Expects a torch tensor with shape Bx4. Requires the original image size in (H, W) format.
    """
    boxes = apply_coords_torch(boxes.reshape(-1, 2, 2), original_size,
                               target_length)
    return boxes.reshape(-1, 4)


transformed_image = prepare_pil_image(img, 1024, sam)
boxes = torch.load('tmp/cell_detection/Embryo0418_bin8_downsampled.pt')
boxes = boxes.to(sam.device)
transformed_boxes = prepare_boxes(boxes, img.size[::-1], 1024)

batched_input = [{
    'image': transformed_image,
    'boxes': transformed_boxes,
    'original_size': img.size[::-1]
}]

batched_output = sam(batched_input, multimask_output=False)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, color='green'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0),
                      w,
                      h,
                      edgecolor=color,
                      facecolor=(0, 0, 0, 0),
                      lw=2))


print(img.size)
print(transforms.PILToTensor()(img).shape)

fig, ax = plt.subplots(1, 2, figsize=(20, 20))

a = 1
if a:
    ax[0].imshow(img)
    for mask in batched_output[0]['masks']:
        show_mask(mask.cpu().numpy(), ax[0], random_color=True)
    for box in boxes:
        show_box(box.cpu().numpy(), ax[0])
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()
