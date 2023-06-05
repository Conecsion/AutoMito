import sys
import os
from PIL import Image
import torch.multiprocessing as mp


def fn(rank, args):
    print(rank)
    print(args)


if __name__ == "__main__":
    args = ("hello", "world")
    mp.spawn(fn, args=(args, ), nprocs=1)
