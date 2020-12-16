#Import necessarry tools from torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn as nn

#Import necessarry tools from tpstorch 
from tpstorch.ml.data import EXPReweightSimulation, TSTValidation
from tpstorch.ml.optim import UnweightedSGD, EXPReweightSGD
from torch.distributed import distributed_c10d 
from mullerbrown import CommittorNet, MullerBrown, MullerBrownLoss
import numpy as np

dist.init_process_group(backend='mpi')
mpi_group = dist.distributed_c10d._get_default_group()

#Import any other thing
import tqdm, sys

prefix = 'simple'

#Initialize neural net
committor = CommittorNet(d=2,num_nodes=200).to('cpu')

#Set initial configuration and BP simulator
start = torch.tensor([[0.0,0.0]])
end = torch.tensor([[1.0,1.0]])
def initializer(s):
    return (1-s)*start+s*end
initial_config = initializer(dist.get_rank()/(dist.get_world_size()-1))
mb_sim = MullerBrown(param="param",config=initial_config, rank=dist.get_rank(), dump=1, beta=1, kappa=1, save_config=True, mpi_group = mpi_group, committor=committor)

#Committor Loss
initloss = nn.MSELoss()
initoptimizer = UnweightedSGD(committor.parameters(), lr=1e-2)#,momentum=0.9,nesterov=True)#, weight_decay=1e-3)

#from torchsummary import summary
running_loss = 0.0
#Initial training try to fit the committor to the initial condition
for i in tqdm.tqdm(range(10**3)):
    # zero the parameter gradients
    initoptimizer.zero_grad()
    # forward + backward + optimize
    q_vals = committor(initial_config)
    targets = torch.ones_like(q_vals)*dist.get_rank()/(dist.get_world_size()-1)
    cost = initloss(q_vals, targets)#,committor,config,cx)
    cost.backward()
    #Stepping up
    initoptimizer.step()
    #committor.renormalize()
    committor.project()
committor.zero_grad()

from torch.optim import lr_scheduler

#Construct EXPReweightSimulation
batch_size = 128
dataset = EXPReweightSimulation(mb_sim, committor, period=10)
loader = DataLoader(dataset,batch_size=batch_size)
