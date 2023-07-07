import cv2
import os
import csv
import concurrent.futures
import multiprocessing
import numpy as np
import time
from typing import Tuple
"""
First determine the minimal size of all input images
Pad the images to the same size, so that the whole image can be cropped into integer numbers of patches
Crop the images according to crop_size and overlap ratio
"""


def check_size(image, queue):
    im = cv2.imread(image)
    size = im.shape[:2]
    queue.put(size)


def crop(crop_size, input_path, output_path, overlap=0.2, chunksize=1):
    os.makedirs(output_path, exist_ok=True)
    print("---------------IMAGE CROPPING---------------")
    time.sleep(0.5)
    img_list = [os.path.join(input_path, s) for s in os.listdir(input_path)]

    # Expand the image to the integer multiple of cropped size
    # Get the expanded image size after padding
    q = multiprocessing.Manager().Queue()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        q_ = [q] * len(img_list)
        executor.map(check_size, img_list, q_)
    q.put("DONE")
    sizes = []
    while True:
        size = q.get()
        if size == "DONE":
            break
        else:
            sizes.append(size)

    min_size = min([min(s) for s in sizes])
    # print(min_size)
    if crop_size > min_size:
        print("ERROR: crop_size should be smaller than image size")
    else:
        step = int(crop_size * (1 - overlap))
        expanded_height = max([(np.ceil(
            (size[0] - crop_size) / step) * step) + crop_size
                               for size in sizes])
        expanded_width = max([(np.ceil(
            (size[1] - crop_size) / step) * step) + crop_size
                              for size in sizes])

        expanded_size = (int(expanded_height), int(expanded_width))

        # Initialize queue
        queue = multiprocessing.Manager().Queue()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            output = [output_path] * len(img_list)
            crop_size = [crop_size] * len(img_list)
            expanded_size = [expanded_size] * len(img_list)
            overlap = [overlap] * len(img_list)
            queue_list = [queue] * len(img_list)
            futures = executor.map(
                crop_one,
                img_list,
                output,
                crop_size,
                expanded_size,
                overlap,
                queue_list,
                chunksize=chunksize,
            )
        queue.put("DONE")
        with open(os.path.join(output_path, 'crop_index.csv'), 'a') as f:
            writer = csv.writer(f)
            while True:
                index_line = queue.get()
                if index_line == "DONE":
                    break
                else:
                    writer.writerow(index_line)

        print("---------------IMAGE CROPPING DONE---------------")


def crop_one(
        img_name: str,
        output_path: str,
        crop_size: int,
        expanded_size: Tuple[int, int],  # (height, width)
        overlap: float,
        queue):
    img = cv2.imread(img_name)
    # print(img.shape)
    # print(expanded_size)

    height, width = img.shape[:2]

    # Padding
    pad_height_before = (expanded_size[0] - height) // 2
    pad_height_after = expanded_size[0] - height - pad_height_before
    pad_width_before = (expanded_size[1] - width) // 2
    pad_width_after = expanded_size[1] - width - pad_width_before

    padded_img = np.pad(img, ((pad_height_before, pad_height_after),
                              (pad_width_before, pad_width_after), (0, 0)),
                        mode='constant',
                        constant_values=0)
    # print(padded_img.shape)

    step = int(crop_size * (1 - overlap))

    for i in range(0, height, step):
        for j in range(0, width, step):
            # print(j, j + crop_size)
            x1 = j
            x2 = min(j + crop_size, padded_img.shape[1])
            # print(f"x2={x2}")
            y1 = i
            y2 = min(i + crop_size, padded_img.shape[0])

            # print(x1, x2)
            patch = padded_img[y1:y2, x1:x2]

            # print(i, j, patch.shape)
            i_n = i // step
            j_n = j // step
            img_base_name = os.path.basename(os.path.splitext(img_name)[0])
            filename = f"{img_base_name}_{i_n}_{j_n}.tif"
            cv2.imwrite(filename, patch)
            index_data = (filename, img_name, (i, j, i + crop_size,
                                               j + crop_size))
            queue.put(index_data)
            print(f"{filename} cropping done")

            if j + crop_size > width:
                break

        if i + crop_size > height:
            break
