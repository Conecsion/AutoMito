#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import concurrent.futures


def getBasename(filename):
    basename = os.path.splitext(filename)[0]
    basename = os.path.basename(basename)
    return basename


def compare_one(file1, file2):
    file1, file2 = map(getBasename, [file1, file2])
    file2 = file2[:-5]
    if not file1 == file2:
        print(file1)


def check_exist(file):
    if not os.path.isfile(file):
        print(file)


with concurrent.futures.ProcessPoolExecutor() as executor:
    file2_list = []
    for dirpath, _, filenames in os.walk("tmp/yolo"):
        for file in filenames:
            file2 = f"{getBasename(file)}_mask.tif"
            file2 = os.path.join("tmp/sam", file2)
            file2_list.append(file2)

    executor.map(check_exist, file2_list)
