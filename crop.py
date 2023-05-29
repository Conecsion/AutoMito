#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import os
from multiprocessing import Pool
import concurrent.futures
import time


def crop(size=1024, input_dir='input', output_dir="output/cropped_images"):
    os.makedirs(output_dir, exist_ok=True)
    if os.listdir(output_dir):
        print(
            "The Output directory of cropping is not empty. Skipping Cropping."
        )
        time.sleep(0.5)
    else:
        print("---------------IMAGE CROPPING---------------")
        time.sleep(0.5)
        img_list = []
        for dirpath, _, filenames in os.walk(input_dir):
            for file in filenames:
                img_list.append(os.path.join(dirpath, file))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            output_list = [output_dir] * len(img_list)
            size_list = [size] * len(img_list)
            executor.map(crop_one, img_list, output_list, size_list)
        print("---------------IMAGE CROPPING DONE---------------")


def crop_one(img_name, output_dir, size):
    img = Image.open(img_name)
    width, height = img.size
    # Expand the image to the integer multiple of cropped size
    expanded_width = (width // size + 1) * size
    expanded_height = (height // size + 1) * size
    pad_width = (expanded_width - width) // 2
    pad_height = (expanded_height - height) // 2
    expanded_img = Image.new("RGB", (expanded_width, expanded_height), "black")
    expanded_img.paste(img, (pad_width, pad_height))

    for i in range(0, expanded_height, size):
        for j in range(0, expanded_width, size):
            area = (j, i, j + size, i + size)
            cropped_img = expanded_img.crop(area)
            filename = os.path.splitext(os.path.basename(img.filename))[0]
            cropped_img.save(
                f"{output_dir}/{filename}_{int(i/size)}_{int(j/size)}.tif")
