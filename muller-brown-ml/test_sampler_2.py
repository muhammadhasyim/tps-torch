import mullerbrown_ml as mb
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d 

dist.init_process_group(backend='mpi')
mpi_group = dist.distributed_c10d._get_default_group()

a = mb.MySampler('param_tst',torch.tensor([[0.5,0.5]]),0,0,2.0,0.0,distributed_c10d._get_default_group())
for i in range(100000):
    config = torch.tensor([[0.0,0.0]])
    a.propose(config, 0, False)
    a.acceptReject(config, 0, False, False)
    test = a.getConfig()
    print(test)
