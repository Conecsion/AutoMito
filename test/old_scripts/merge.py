#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import os
from multiprocessing import Pool
import concurrent.futures

output_path = '/home/shaodi/Work/Embryo/yolov8/output/predict7'
input_path = '/home/shaodi/Work/Embryo/yolov8/runs/detect/predict7'


def merge(img_index):
    size = 512
    merged_img = Image.new('RGB', (8192, 8192))
    for i in range(16):
        for j in range(16):
            img_name = str(img_index) + '_' + str(i) + '_' + str(j) + '.png'
            img_name = os.path.join(input_path, img_name)
            cropped_img = Image.open(img_name)
            merged_img.paste(cropped_img, (j * size, i * size))
    merged_img_name = str(img_index) + '_det.png'

    merged_img.save(os.path.join(output_path, merged_img_name))


if __name__ == '__main__':
    index = range(321, 1440 + 1)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(merge, index)
