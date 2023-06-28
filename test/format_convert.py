#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import os
from multiprocessing import Pool
import concurrent.futures
import time
import sys


# Convert all TIFF images in the input_dir to PNG format or vice versa.
def format_convert(input_dir, output_dir):
    Image.MAX_IMAGE_PIXELS = None
    os.makedirs(output_dir, exist_ok=True)
    if os.listdir(output_dir):
        print(
            "The Output directory of format conversion is not empty. Skipping format conversion."
        )
        time.sleep(0.5)
    else:
        print("---------------FORMAT CONVERTING---------------")
        time.sleep(0.5)
        img_list = []
        for dirpath, _, filenames in os.walk(input_dir):
            for file in filenames:
                img_list.append(os.path.join(dirpath, file))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            output_list = [output_dir] * len(img_list)
            executor.map(convert_one, img_list, output_list)
        print("---------------FORMAT CONVERSION DONE---------------")


def convert_one(img, output_dir):
    img = Image.open(img)
    filename = os.path.basename(img.filename)
    basename, ext = os.path.splitext(filename)
    if ext == ".tif" or ext == ".tiff":
        target_format = ".png"
    elif ext == ".png":
        target_format = ".tif"
    img.save(os.path.join(output_dir, f"{basename}{target_format}"))
