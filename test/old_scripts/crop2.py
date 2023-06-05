#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Used to crop a large image to size 512 * 512 and write the cropped images to the output_dir
# Usage: ./crop.py input_img output_dir


from PIL import Image
import argparse 
import os


parser = argparse.ArgumentParser()
parser.add_argument('input_img')
parser.add_argument('output_dir')
args = parser.parse_args()


size = 512 

img = Image.open(args.input_img)
filename = os.path.basename(os.path.splitext(args.input_img)[0])
width, height = img.size 

for i in range(0, height, size):
    for j in range(0, width, size):
        area = (j, i, j+size, i+size)
        cropped_img = img.crop(area)
        cropped_img.save(f"{args.output_dir}/{filename}_{int(i/size)}_{int(j/size)}.png")
