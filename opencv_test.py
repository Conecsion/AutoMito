#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from segment_anything.utils.transforms import ResizeLongestSide
import torch
from torchvision.transforms import ToPILImage, ToTensor
import numpy as np
from segment_anything import sam_model_registry
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Tuple
import cv2

torch.cuda.set_device(1)

sam = sam_model_registry['vit_h'](checkpoint='model/sam_vit_h_4b8939.pth')
sam.to(device='cuda')

resize_transform = ResizeLongestSide(sam.image_encoder.img_size)


def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)
    return image.permute(2, 0, 1).contiguous()


# Return the shape (H, W) of the target image/boxes after preprocessing from the old shape
#  def get_preprocess_shape(oldh: int, oldw: int,
#  target_length: int) -> Tuple[int, int]:
#  scale = target_length * 1.0 / max(oldh, oldw)
#  newh, neww = oldh * scale, oldw * scale
#  neww = int(neww + 0.5)
#  newh = int(newh + 0.5)
#  return (newh, neww)

# Resize a PIL Image with target length
#  def prepare_image(image: Image.Image, target_length: int) -> Image.Image:
#  target_size = get_preprocess_shape(image.size[0], image.size[1],
#  target_length)
#  return resize(image, target_size)
#
#  return image.thumbnail(target_size, Image.BILINEAR)
#
#
#  def apply_coords_torch(coords: torch.Tensor, original_size: Tuple[int, ...],
#  target_length: int) -> torch.Tensor:
#  """
#  Expects a torch tensor with length 2 in the last dimension. Requires the
#  original image size in (H, W) format.
#  """
#  old_h, old_w = original_size
#  new_h, new_w = get_preprocess_shape(original_size[0], original_size[1],
#  target_length)
#  coords = deepcopy(coords).to(torch.float)
#  coords[..., 0] = coords[..., 0] * (new_w / old_w)
#  coords[..., 1] = coords[..., 1] * (new_h / old_h)
#  return coords
#
#
#  def apply_boxes_torch(boxes: torch.Tensor, original_size: Tuple[int, ...],
#  target_length: int) -> torch.Tensor:
#  """
#  Expects a torch tensor with shape Bx4. Requires the original image
#  size in (H, W) format.
#  """
#  boxes = apply_coords_torch(boxes.reshape(-1, 2, 2), original_size,
#  target_length)
#  return boxes.reshape(-1, 4)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0),
                      w,
                      h,
                      edgecolor='green',
                      facecolor=(0, 0, 0, 0),
                      lw=2))


#  im = Image.open('tmp/downsampled_img/Embryo0418_bin8_downsampled.tif').convert(
#  'RGB')

im = cv2.imread('tmp/downsampled_img/Embryo0418_bin8_downsampled.tif',
                cv2.IMREAD_GRAYSCALE)
print(im.shape)
im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
print(im.shape)

#  width, height = im.size
#  expanded_size = (1436, 1436)
#  pad_width = (expanded_size[0] - width) // 2
#  pad_height = (expanded_size[1] - height) // 2
#  expanded_img = Image.new("RGB", expanded_size, "black")
#  expanded_img.paste(im, (pad_width, pad_height))
#
#  im = expanded_img
boxes = torch.load('tmp/cell_detection/Embryo0418_bin8_downsampled.pt')
boxes = boxes.to(sam.device)
#  boxes_cuda = apply_boxes_torch(boxes, im.size, 1024)
#  boxes_cuda = boxes.to('cuda')
print((prepare_image(im, resize_transform, sam).dtype))
print((prepare_image(im, resize_transform, sam).shape))

batched_input = [{
    'image':
    prepare_image(im, resize_transform, sam),
    'boxes':
    resize_transform.apply_boxes_torch(boxes, im.shape[:2]),
    'original_size':
    im.shape[:2]
}]

batched_output = sam(batched_input, multimask_output=False)

fig, ax = plt.subplots(1, 2, figsize=(20, 20))

a = 0
if a:
    ax[0].imshow(im)
    for mask in batched_output[0]['masks']:
        show_mask(mask.cpu().numpy(), ax[0], random_color=True)
    for box in boxes:
        show_box(box.cpu().numpy(), ax[0])
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()
