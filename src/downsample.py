#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script is used to downsample the whole embryo image,
# in order to segment the area of the two cells in the image

from PIL import Image
import os
import sys
import glob
import concurrent.futures

sys.path.append("..")


# Downsample all images in the input_path
def downsample(bin_factor=8,
               input_path='input',
               output_path='tmp/downsampled_img',
               output_format='tif'):
    Image.MAX_IMAGE_PIXELS = None
    os.makedirs(output_path, exist_ok=True)
    if os.listdir(output_path):
        if os.listdir(output_path):
            choice = input(
                f'Output directory of downsampling is not empty. Do you want to delete the existed files in {output_path}? (y/N):'
            )
            if choice == 'y' or choice == 'Y':
                files = os.listdir(output_path)
                for file in files:
                    os.remove(os.path.join(output_path, file))
    print("---------------IMAGE DOWNSAMPLING---------------")
    img_list = [os.path.join(input_path, s) for s in os.listdir(input_path)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        output_list = [output_path] * len(img_list)
        bin_list = [bin_factor] * len(img_list)
        format_list = [output_format] * len(img_list)
        executor.map(downsample_one, img_list, output_list, bin_list,
                     format_list)
    print("---------------IMAGE DOWNSAMPLING DONE---------------")


def downsample_one(img, output_path, bin_factor, output_format):
    img = Image.open(img)
    downsampled = img.resize(
        (img.width // bin_factor, img.height // bin_factor), Image.BILINEAR)
    filename = os.path.basename(os.path.splitext(img.filename)[0])
    downsampled.save(
        os.path.join(
            output_path,
            f'{filename}_bin{bin_factor}_downsampled.{output_format}'))
    print(f'{filename} downsample done')


def upsample_one_mask(downsampled_mask_path, output_path, original_image: str):
    original_image = Image.open(original_image)
    original_size = original_image.size
    filename = os.path.basename(os.path.splitext(original_image.filename)[0])
    mask_name = glob.glob(
        os.path.join(downsampled_mask_path, filename) + '_*')[0]
    mask = Image.open(mask_name)
    upsampled = mask.resize(original_size, Image.Resampling.BILINEAR)
    upsampled.save(os.path.join(output_path, f'{filename}_mask.tif'))
    print(f'{filename}_mask upsample done')


def upsample_masks(input_path='tmp/downsampled_cell_mask',
                   original_image_path='input',
                   output_path='output/cell_mask'):
    Image.MAX_IMAGE_PIXELS = None
    os.makedirs(output_path, exist_ok=True)
    if os.listdir(output_path):
        choice = input(
            f'Output directory of upsampling is not empty. Do you want to delete the existed files in {output_path}? (y/N):'
        )
        if choice == 'y' or choice == 'Y':
            files = os.listdir(output_path)
            for file in files:
                os.remove(os.path.join(output_path, file))
    print('---------------UPSAMPLING---------------')
    original_image_list = [
        os.path.join(original_image_path, s)
        for s in os.listdir(original_image_path)
    ]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(
            upsample_one_mask,
            [input_path] * len(original_image_list),
            [output_path] * len(original_image_list),
            original_image_list,
        )
    print('---------------UPSAMPLING DONE---------------')


if __name__ == '__main__':
    downsample()
