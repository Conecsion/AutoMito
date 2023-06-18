#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Merge masks from cropped size to whole size
#
# The raw images have been padded while cropping, thus need to be re-merged too

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
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


def chunknizer(mylist, chunk_size):
    chunk_list = []
    for i in range(0, len(mylist), chunk_size):
        chunk = mylist[i:i + chunk_size]
        chunk_list.append(chunk)

    return chunk_list


# Generate empty dark mask images for the patches which have no detection
def generate_one_blank_mask(one_mask_path, crop_size):
    if not os.path.isfile(one_mask_path):
        blank_mask = Image.new('L', (crop_size, crop_size), color=0)
        blank_mask.save(one_mask_path)
        print(f'Generation of {getOnlyname(one_mask_path)} blank mask done')


def generate_blank_masks(cropped_img_path, mask_path, crop_size):
    masks = []
    for s in os.listdir(cropped_img_path):
        onlyname = getOnlyname(s)
        mask = os.path.join(mask_path, f'{onlyname}_mask.tif')
        masks.append(mask)

    chunk_masks_list = chunknizer(masks, 10000)
    for masks in chunk_masks_list:
        with concurrent.futures.ProcessPoolExecutor() as mask_gen_executor:
            mask_gen_executor.map(generate_one_blank_mask, masks,
                                  [crop_size] * len(masks))
    print('Blank masks generation done')


def merge_one(input_img, cropped_img_path, mask_path, crop_size,
              img_output_path, mask_output_path):
    img_name = getOnlyname(input_img)
    cropped_img_list = glob.glob(
        os.path.join(cropped_img_path, f"{img_name}_*"))
    mask_list = glob.glob(os.path.join(mask_path, f"{img_name}_*"))

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

        merged_img = merged_img.convert('L')
        merged_img.save(
            os.path.join(img_output_path, f'{img_name}_remerge.tif'))
        merged_mask.save(os.path.join(mask_output_path,
                                      f'{img_name}_mask.tif'))
        print(f'{img_name} merged')


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
    chunk_input_imgs = chunknizer(input_imgs, 100)
    img_num = len(chunk_input_imgs)
    cropped_img_num = len(os.listdir(cropped_img_path))
    mask_num = len(os.listdir(mask_path))
    if cropped_img_num != mask_num:
        print('ERROR: The number of cropped images and masks is not match')
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(merge_one, chunk_input_imgs,
                         img_num * [cropped_img_path], [mask_path] * img_num,
                         [crop_size] * img_num)


#  generate_one_blank_mask("tmp/Embryo315_13_1_mask.tif", 512)
