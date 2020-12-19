#Start up torch dist package
import torch
import torch.distributed as dist
dist.init_process_group(backend='mpi')

#Load classes for simulations and controls
from brownian_fts import BrownianParticle, CustomFTSMethod
import numpy as np

#Starting and ending configuration.
start = torch.tensor([[-1.0]])
end = torch.tensor([[1.0]])
def initializer(s):
    return (1-s)*start+s*end
alphas = torch.linspace(0.0,1,dist.get_world_size()+2)[1:-1]

bp_simulator = BrownianParticle(dt=2e-3,gamma=1.0, kT = 0.4, initial=initializer(alphas[dist.get_rank()]),save_config=True)
fts_method = CustomFTSMethod(sampler=bp_simulator,initial_config=start,final_config=end,num_nodes=dist.get_world_size()+2,deltatau=0.01,kappa=0.01)

import tqdm
for i in tqdm.tqdm(range(400000)):
    #Run the simulation a single time-step
    fts_method.run(1)
    if (i % 10 == 0):
        fts_method.dump(True)
