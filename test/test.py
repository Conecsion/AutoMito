import sys
import os
from PIL import Image
import torch.multiprocessing as mp

print(os.getcwd())
from format_convert import format_convert

if __name__ == "__main__":
    format_convert('project/project3/crop/', 'project/png/')
