#!/usr/bin/env python
# -*- coding: utf-8 -*-


# This script is used to crop one input image to the size and coordinate set by "--box" option
# Can be used to crop a small size from a large image for test

from PIL import Image 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
parser.add_argument('-b','--box',nargs=4, type=int)

args = parser.parse_args()


img = Image.open(args.input)
box = tuple(args.box)
cropped_img = img.crop(box)
cropped_img.save(args.output)
