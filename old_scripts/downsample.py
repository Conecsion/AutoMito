#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from PIL import Image
import argparse 


parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()


if __name__ == '__main__':
    size = 1024, 1024

    with Image.open(args.input) as img:
        img.thumbnail(size)
        img.save(args.output, "PNG")
