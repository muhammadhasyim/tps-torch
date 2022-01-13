#import sys
#sys.path.insert(0,"/global/home/users/muhammad_hasyim/tps-torch/build")

#Import necessarry tools from torch
import tpstorch
import torch
import torch.distributed as dist
import torch.nn as nn

#Import necessarry tools from tpstorch 
from dimer_ftsus import FTSLayerUSCustom as FTSLayer
from committor_nn import CommittorNetDR
#from tpstorch.ml.data import FTSSimulation, EXPReweightStringSimulation
from tpstorch.ml.optim import ParallelSGD, ParallelAdam#, FTSImplicitUpdate, FTSUpdate
#from tpstorch.ml.nn import BKELossFTS, BKELossEXP, FTSCommittorLoss, FTSLayer
import numpy as np

#Grag the MPI group in tpstorch
mpi_group = tpstorch._mpi_group
world_size = tpstorch._world_size
rank = tpstorch._rank

#Import any other thing
import tqdm, sys
torch.manual_seed(5070)
np.random.seed(5070)

prefix = 'simple'

#Initialize neural net
def initializer(s):
    return (1-s)*start+s*end

#Initialization
r0 = 2**(1/6.0)
width =  0.5*r0

#Reactant
dist_init = r0-0.20*r0
start = torch.zeros((2,3))
start[0][2] = -0.5*dist_init
start[1][2] = 0.5*dist_init

#Product state
dist_init = r0+2*width+0.20*r0
end = torch.zeros((2,3))
end[0][2] = -0.5*dist_init
end[1][2] = 0.5*dist_init

if rank == 0:
    print("At NN start stuff")

committor = CommittorNetDR(num_nodes=50, boxsize=10).to('cpu')
ftslayer = FTSLayer(react_config=start.flatten(),prod_config=end.flatten(),num_nodes=world_size,boxsize=10.0,kappa_perpend=0.0,kappa_parallel=0.0).to('cpu')

#Initial Training Loss
initloss = nn.MSELoss()
initoptimizer = ParallelAdam(committor.parameters(), lr=1e-3)

#from torchsummary import summary
running_loss = 0.0
tolerance = 1e-3

#Initial training try to fit the committor to the initial condition
tolerance = 1e-4
#batch_sizes = [64]
#for size in batch_sizes:
loss_io = []
if rank == 0:
    loss_io = open("{}_statistic_{}.txt".format(prefix,rank+1),'w')
if rank == 0:
    print("Before training")
for i in range(10**7):
    # zero the parameter gradients
    initoptimizer.zero_grad()
    
    # forward + backward + optimize
    q_vals = committor(ftslayer.string[rank])#initial_config.view(-1,2))
    targets = torch.ones_like(q_vals)*rank/(dist.get_world_size()-1)
    cost = initloss(q_vals, targets)#,committor,config,cx)
    cost.backward()
    
    #Stepping up
    initoptimizer.step()
    with torch.no_grad():
        dist.all_reduce(cost)
        #if i % 10 == 0 and rank == 0:
        #    print(i,cost.item() / world_size, committor(ftslayer.string[-1]))
        #    torch.save(committor.state_dict(), "initial_1hl_nn")#_{}".format(size))#prefix,rank+1))
        if rank == 0:
            loss_io.write("Step {:d} Loss {:.5E}\n".format(i,cost.item()))
            loss_io.flush()
        if cost.item() / world_size < tolerance:
            if rank == 0:
                torch.save(committor.state_dict(), "initial_1hl_nn")#_{}".format(size))#prefix,rank+1))
                torch.save(ftslayer.state_dict(), "test_string_config")#_{}".format(size))#prefix,rank+1))
            print("Early Break!")
            break
    committor.zero_grad()
