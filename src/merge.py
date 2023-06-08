#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Merge masks from cropped size to whole size
#
# The raw images have been padded while cropping, thus need to be re-merged too

from PIL import Image
import os
import torch
import concurrent.futures
import glob


# Generate empty dark mask images for the patches which have no detection
def generate_one_blank_mask(mask_path, crop_size):
    if not os.path.isfile(mask_path):
        blank_mask = Image.new('L', (crop_size, crop_size))
        blank_mask.paste(0, (0, 0, blank_mask.width, blank_mask.height))
        blank_mask.save(mask_path)


def generate_blank_masks(masks_path, crop_size):
    masks = os.listdir(masks_path)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(generate_blank_masks, masks, [crop_size] * len(masks))


def merge_one(cropped_imgs_sublist, masks_sublist, crop_size, output_path):
    img_name = getOnlyname(cropped_imgs_sublist[0]).split('_')[:-2]
    img_name = f'{img_name}.tif'
    mask_name = f'{img_name}_mask.tif'
    x_list = []
    y_list = []
    for img in cropped_imgs_sublist:
        y, x = parse_numbers(img)
        x_list.append(x)
        y_list.append(y)
    width = max(x_list) * crop_size
    height = max(y_list) * crop_size
    merged_img = Image.new('RGB', (width, height))
    merged_mask = Image.new('L', (width, height))
    for i in range(len(cropped_imgs_sublist)):
        img = Image.open(cropped_imgs_sublist[i])
        mask = Image.open(masks_sublist[i])
        x_coord = crop_size * x_list[i]
        y_coord = crop_size * y_list[i]
        merged_img.paste(img, (x_coord, y_coord))
        merged_mask.paste(mask, (x_coord, y_coord))
    print("Saving merged {img_name}")
    merged_img.save(os.path.join(output_path, "images", img_name))
    merged_mask.save(os.path.join(output_path, "masks", mask_name))


def merge_all(cropped_imgs_list, masks_list, crop_size, output_path):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(merge_one, cropped_imgs_list, masks_list,
                     [crop_size] * len(cropped_imgs_list),
                     [output_path] * len(cropped_imgs_list))


def parse_numbers(filename, item="img"):
    filename = getOnlyname(filename)
    if item == "img":
        numbers = filename.split("_")[-2:]
        numbers = [int(s) for s in numbers]
        return numbers
    elif item == "mask":
        numbers = filename.split("_")[-3:-1]
        numbers = [int(s) for s in numbers]
        return numbers


def getOnlyname(wholename):
    onlyname = os.path.splitext(wholename)[0]
    onlyname = os.path.basename(onlyname)
    return onlyname


def merge(crop_size=512,
          input_path="input",
          cropped_img_path="tmp/crop",
          yolo_path="tmp/yolo",
          mask_path="tmp/sam",
          output_path="output"):

    generate_blank_masks(mask_path, crop_size)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "masks"), exist_ok=True)

    # Get cropped images list and masks list
    input_imgs = os.listdir(input_path)
    cropped_imgs = os.listdir(cropped_img_path)
    cropped_imgs_list = []
    masks_list = []
    for file in input_imgs:
        file = getOnlyname(file)
        cropped_imgs_sublist = [s for s in cropped_imgs if file in s]
        cropped_imgs_sublist.sort(key=parse_numbers)
        masks_sublist = [
            f"{getOnlyname(s)}_mask.tif" for s in cropped_imgs_sublist
        ]
        cropped_imgs_sublist = [
            os.path.join(cropped_img_path, s) for s in cropped_imgs_sublist
        ]
        masks_sublist = [os.path.join(mask_path, s) for s in masks_sublist]

        cropped_imgs_list.append(cropped_imgs_sublist)
        masks_list.append(masks_sublist)

    merge_one(cropped_imgs_list[0], masks_list[0], 512, "output")
    #  merge_all(cropped_imgs_list, masks_list, crop_size, output_path)


if __name__ == "__main__":
    merge()
