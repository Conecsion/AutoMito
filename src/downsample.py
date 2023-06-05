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


# Downsample all images in the input_dir
def downsample(bin=8, input_dir='input', output_dir='downsampled_img'):
    Image.MAX_IMAGE_PIXELS = None
    os.makedirs(output_dir, exist_ok=True)
    if os.listdir(output_dir):
        print(
            "The Output directory of downsampling is not empty. Skip downsampling."
        )
    else:
        print("---------------IMAGE DOWNSAMPLING---------------")
        img_list = []
        for dirpath, _, filenames in os.walk(input_dir):
            for file in filenames:
                img_list.append(os.path.join(dirpath, file))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            output_list = [output_dir] * len(img_list)
            bin_list = [bin] * len(img_list)
            executor.map(downsample_one, img_list, output_list, bin_list)
        print("---------------IMAGE DOWNSAMPLING DONE---------------")


def downsample_one(img, output_dir, bin):
    img = Image.open(img)
    downsampled = img.resize((img.width // bin, img.height // bin),
                             Image.BILINEAR)
    filename = os.path.basename(os.path.splitext(img.filename)[0])
    downsampled.save(f"{output_dir}/{filename}_bin{bin}_downsampled.tif")
