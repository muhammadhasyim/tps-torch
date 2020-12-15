import mullerbrown_ml as mb
import torch
import torch.distributed as dist

dist.init_process_group(backend='mpi')
mpi_group = dist.distributed_c10d._get_default_group()

a = mb.MySampler('param',torch.tensor([[0.0,0.0]]),0,0,0.0,0.0,dist.distributed_c10d._get_default_group())
