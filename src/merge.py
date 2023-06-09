#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Merge masks from cropped size to whole size
#
# The raw images have been padded while cropping, thus need to be re-merged too

from PIL import Image
import os
import concurrent.futures
import glob


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


# Generate empty dark mask images for the patches which have no detection
def generate_one_blank_mask(one_mask_path, crop_size):
    if not os.path.isfile(one_mask_path):
        print(f'Generating blank mask for {getOnlyname(one_mask_path)}')
        blank_mask = Image.new('L', (crop_size, crop_size), color=0)
        blank_mask.save(one_mask_path)
        print('Generation of blank mask done')


def generate_blank_masks(cropped_img_path, mask_path, crop_size):
    masks = []
    for s in os.listdir(cropped_img_path):
        onlyname = getOnlyname(s)
        mask = os.path.join(mask_path, f'{onlyname}_mask.tif')
        masks.append(mask)
    with concurrent.futures.ProcessPoolExecutor() as mask_gen_executor:
        mask_gen_executor.map(generate_one_blank_mask, masks,
                              [crop_size] * len(masks))
        print('Blank masks generation done')


def merge_one(input_img, cropped_img_path, mask_path, crop_size):
    img_name = getOnlyname(input_img)
    cropped_img_list = glob.glob(os.path.join(cropped_img_path,
                                              f"{img_name}_"))
    mask_list = glob.glob(os.path.join(mask_path, f"{img_name}_"))

    img_coord_list = []
    for s in cropped_img_list:
        img_coord_list.append(parse_numbers(s, 'img'))
    mask_coord_list = []
    for s in mask_list:
        mask_coord_list.append(parse_numbers(s, 'mask'))
    if len(img_coord_list) != len(mask_coord_list):
        print(f'ERROR: {img_name} cropped images and masks not match')
        return 0
    else:
        img_size = Image.open(input_img).size
        merged_img = Image.new('RGB', img_size)
        merged_mask = Image.new('L', img_size)
        for i in range(len(img_coord_list)):
            img = Image.open(cropped_img_list[i])
            img_x, img_y = crop_size * img_coord_list[i][
                1], crop_size * img_coord_list[i][0]
            mask = Image.open(mask_list[i])
            mask_x, mask_y = crop_size * mask_coord_list[i][
                1], crop_size * mask_coord_list[i][0]
            merged_img.paste(img, (img_x, img_y))
            merged_mask.paste(mask, (mask_x, mask_y))


def merge(crop_size=512,
          input_path="input",
          output_path="output",
          cropped_img_path="tmp/crop",
          yolo_path="tmp/yolo",
          mask_path="tmp/sam"):

    generate_blank_masks(cropped_img_path, mask_path, crop_size)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "masks"), exist_ok=True)

    # Get cropped images list and masks list
    input_imgs = [os.path.join(input_path, s) for s in os.listdir(input_path)]
    img_num = len(input_imgs)
    cropped_img_num = len(os.listdir(cropped_img_path))
    mask_num = len(os.listdir(mask_path))
    if cropped_img_num != mask_num:
        print('ERROR: The number of cropped images and masks is not match')
    else:
        print(input_imgs)
        print(cropped_img_path)
        print(mask_path)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(merge_one, input_imgs, img_num * [cropped_img_path],
                         [mask_path] * img_num, [crop_size] * img_num)


#  generate_one_blank_mask("tmp/Embryo315_13_1_mask.tif", 512)
#  generate_blank_masks("tmp/crop", "tmp/sam", 512)
