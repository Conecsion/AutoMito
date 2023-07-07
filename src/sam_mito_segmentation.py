#!/usr/bin/env python
# -*- coding: utf-8 -*-
from segment_anything.utils.transforms import ResizeLongestSide
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from segment_anything import sam_model_registry
import pandas as pd
import numpy as np

# Import DDP related packages
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os


def get_preprocess_shape(oldh: int, oldw: int,
                         long_side_length: int) -> tuple[int, int]:
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


#  resize_transform = ResizeLongestSide(sam.image_encoder.img)


def prepare_image(image: Image.Image, sam) -> torch.Tensor:
    image = np.asarray(image)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    image = resize_transform.apply_image(image)
    image = torch.as_tensor(image, device=sam.device)
    return image.permute(2, 0, 1).contiguous()


def prepare_boxes(boxes: torch.Tensor, sam,
                  original_image: Image.Image) -> torch.Tensor:
    original_size = np.asarray(original_image).shape[:2]
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    return resize_transform.apply_boxes_torch(boxes,
                                              original_size).to(sam.device)


class SamDataset(Dataset):

    def __init__(self, csv_file, model):
        self.dataframe = pd.read_csv(csv_file)
        self.model = model

    def __getitem__(self, idx):
        #  resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
        image = Image.open(self.dataframe.iloc[idx, 0]).convert('RGB')
        original_image = image
        image = prepare_image(image, self.model)
        image.requires_grad = False
        label = torch.load(self.dataframe.iloc[idx, 1])
        label = prepare_boxes(label, self.model, original_image)
        label.requires_grad = False

        root, _ = os.path.splitext(self.dataframe.iloc[idx, 0])
        img_name = os.path.basename(root)

        return [
            dict(
                zip(("image", "boxes", "original_size"),
                    (image, label, image.shape[1:]))), img_name
        ]

    def __len__(self):
        return len(self.dataframe)


def collate_fn(data):
    batched_input = []
    batched_imgname = []
    for i in range(len(data)):
        batched_input.append(data[i][0])
        batched_imgname.append(data[i][1])
    return (batched_input, batched_imgname)


def sam_init(device="cuda",
             sam_checkpoint="model/sam_vit_h_4b8939.pth",
             model_type="vit_h"):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam


def ddp(rank, world_size, batch_size, input_path, output_path, sam_checkpoint,
        model_type):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    model = sam_init(rank, sam_checkpoint, model_type)
    ddp_model = DDP(model, device_ids=[rank])

    dataset = SamDataset(os.path.join(input_path, "yolo_prediction.csv"),
                         model)
    dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=dist_sampler,
                            collate_fn=collate_fn)

    total = len(os.listdir(input_path))
    for batched_input in dataloader:
        batched_output = ddp_model(batched_input[0], multimask_output=False)
        batched_imgname = batched_input[1]
        for i in range(len(batched_output)):
            img_name = batched_imgname[i]
            save_path = (os.path.join(output_path, f'{img_name}_mask.tif'))
            if not os.path.isfile(save_path):
                print(f'Segmenting {img_name}')
                masks = batched_output[i]['masks']
                whole_mask = torch.sum(masks, dim=0,
                                       dtype=bool).to(torch.uint8)
                whole_mask = 255 * whole_mask
                mask_img = ToPILImage()(whole_mask)
                mask_img.save(save_path)

    cleanup()


def cleanup():
    dist.destroy_process_group()


def sam_mito_segmentation(gpu_ids='0,1,2,3',
                          batch_size=3,
                          input_path="tmp/yolo",
                          output_path="tmp/seg_crop",
                          sam_checkpoint="model/sam_vit_h_4b8939.pth",
                          model_type="vit_h"):
    os.makedirs(output_path, exist_ok=True)
    world_size = len(gpu_ids.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    mp.spawn(
        ddp,
        args=(
            world_size,
            batch_size,
            input_path,
            output_path,
            sam_checkpoint,
            model_type,
        ),
        nprocs=world_size,
        join=True,
    )
