#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from PIL import Image
from torchvision import transforms
import matplotlib

#  matplotlib.use('Qt5Agg')
#  matplotlib.use('xcb')
import matplotlib.pyplot as plt

import numpy as np

image = Image.open('./Embryo0418_bin8_downsampled.tif')
print(image.size)

x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
