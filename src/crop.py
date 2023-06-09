#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import os
import concurrent.futures
import time
import sys

sys.path.append("..")


def crop(crop_size=512, input_path='input', output_path="tmp/crop"):
    os.makedirs(output_path, exist_ok=True)
    print("---------------IMAGE CROPPING---------------")
    time.sleep(0.5)
    img_list = [os.path.join(input_path, s) for s in os.listdir(input_path)]

    # Expand the image to the integer multiple of cropped size
    # Get the expanded image size after padding
    sizes = [Image.open(im).size for im in img_list]
    expanded_width = max([(s[0] // crop_size + 1) * crop_size for s in sizes])
    expanded_height = max([(s[1] // crop_size + 1) * crop_size for s in sizes])
    expanded_size = (expanded_width, expanded_height)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        output_list = [output_path] * len(img_list)
        crop_size_list = [crop_size] * len(img_list)
        expanded_size_list = [expanded_size] * len(img_list)
        executor.map(crop_one, img_list, output_list, crop_size_list,
                     expanded_size_list)
    print("---------------IMAGE CROPPING DONE---------------")


def crop_one(img_name, output_path, crop_size, expanded_size):
    img = Image.open(img_name)
    width, height = img.size
    pad_width = (expanded_size[0] - width) // 2
    pad_height = (expanded_size[1] - height) // 2
    expanded_img = Image.new("RGB", expanded_size, "black")
    expanded_img.paste(img, (pad_width, pad_height))

    for i in range(0, expanded_size[1], crop_size):
        for j in range(0, expanded_size[0], crop_size):
            filename = os.path.splitext(os.path.basename(img.filename))[0]
            filename = os.path.join(
                output_path,
                f"{filename}_{int(i/crop_size)}_{int(j/crop_size)}.tif")
            if not os.path.isfile(filename):
                area = (j, i, j + crop_size, i + crop_size)
                cropped_img = expanded_img.crop(area)
                # PIL Image automatically save the image as RGB
                cropped_img.save(filename)
