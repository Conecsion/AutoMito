#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import concurrent.futures
import numpy as np
import time
import sys
from typing import Tuple

img = cv2.imread("Embryo1045.tif")

height, width = img.shape[:2]

patch_size = 2048
overlap = 0.2

step = int(patch_size * (1 - overlap))

patches = []
print(height, width)

for i in range(0, height, step):
    for j in range(0, width, step):
        # print(j, j + patch_size)
        x1 = j
        x2 = min(j + patch_size, width)
        # print(f"x2={x2}")
        y1 = i
        y2 = min(i + patch_size, height)

        # print(x1, x2)
        patch = img[y1:y2, x1:x2]
        patch = np.pad(patch, [(0, patch_size - patch.shape[0]),
                               (0, patch_size - patch.shape[1]), (0, 0)],
                       mode='constant')
        # print(x1, x2, y1, y2)
        # print(patch.shape)
        patch.imwrite(filename)
        if j + patch_size > width:
            break

    if i + patch_size > height:
        break
