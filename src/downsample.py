#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script is used to downsample the whole embryo image,
# in order to segment the area of the two cells in the image

from PIL import Image
import os
from multiprocessing import Pool
import concurrent.futures
import sys

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
        executor.map(downsample_one, img_list, output_list, bin_list)
    print("---------------IMAGE DOWNSAMPLING DONE---------------")


def downsample_one(img, output_path, bin_factor, output_format='tif'):
    img = Image.open(img)
    downsampled = img.resize(
        (img.width // bin_factor, img.height // bin_factor), Image.BILINEAR)
    filename = os.path.basename(os.path.splitext(img.filename)[0])
    downsampled.save(
        os.path.join(
            output_path,
            f'{filename}_bin{bin_factor}_downsampled.{output_format}'))
    print(f'{filename} downsample done')


def upsample_one(mask, output_path, bin_factor):
    mask = Image.open(mask)
    upsampled = mask.resize(
        (mask.width * bin_factor, mask.height * bin_factor), Image.BILINEAR)
    filename = os.path.basename(os.path.splitext(mask.filename)[0])
    upsampled.save(
        os.path.join(output_path, f'{filename}_bin{bin_factor}_upsampled.tif'))
    print(f'{filename} upsample done')


def upsample(bin_factor=8,
             input_path='tmp/downsampled_cell_mask',
             output_path='tmp/cell_mask'):
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
    mask_list = [os.path.join(input_path, s) for s in os.listdir(input_path)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(upsample_one, mask_list, [output_path] * len(mask_list),
                     [bin_factor] * len(mask_list))
    print('---------------UPSAMPLING DONE---------------')


if __name__ == '__main__':
    downsample()
