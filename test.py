import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl", world_size=N, init_method="...")
global_rank = dist.get_rank()
local_rank = global_rank % torch.cuda.device_count()
print(global_rank)
print(torch.cuda.device_count())