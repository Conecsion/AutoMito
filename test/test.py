import sys
import os
from PIL import Image
import torch.multiprocessing as mp

if __name__ == "__main__":
    raw = Image.open("input/Embryo0324.tif")
    crop = Image.open("tmp/crop/Embryo0324_10_10.tif")
    print(raw.mode)
    print(crop.mode)
    convert = crop.convert("L")
    print(convert.mode)
