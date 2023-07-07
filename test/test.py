#!/usr/bin/env python
import sys
import os

sys.path.append('.')

# from crop import crop
from src.crop import crop, crop_one

if __name__ == "__main__":
    # crop(2048, 'test_imgs', 'test_output', 0.2)
    # crop_one('./test_imgs/Embryo1045.tif', './test_output', 2048,
    #          (11876, 10238), 0.2)
    crop(2048, 'input', 'output/crop_with_overlap', 0.2, 1)
