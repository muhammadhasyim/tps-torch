#import sys
#sys.path.insert(0,"/global/home/users/muhammad_hasyim/tps-torch/build")

#Import necessarry tools from torch
import tpstorch
import torch
import torch.distributed as dist
import torch.nn as nn

#Import necessarry tools from tpstorch 
from committor_nn import CommittorNetDR, CommittorNetBP
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
Np = 32
prefix = 'simple'

#Initialize neural net
def initializer(s):
    return (1-s)*start+s*end

#Initialization
r0 = 2**(1/6.0)
width =  0.5*r0

#Reactant
dist_init_start = r0
#Product state
dist_init_end = r0+2*width

#scale down/up the distance of one of the particle dimer
box = [14.736125994561544, 14.736125994561544, 14.736125994561544]
def CubicLattice(dist_init):
    state = torch.zeros(Np, 3);
    num_spacing = np.ceil(Np**(1/3.0))
    spacing_x = box[0]/num_spacing;
    spacing_y = box[1]/num_spacing;
    spacing_z = box[2]/num_spacing;
    count = 0;
    id_x = 0;
    id_y = 0;
    id_z = 0;
    while Np > count:
        state[int(id_z+id_y*num_spacing+id_x*num_spacing*num_spacing)][0] = spacing_x*id_x-0.5*box[0];
        state[int(id_z+id_y*num_spacing+id_x*num_spacing*num_spacing)][1] = spacing_y*id_y-0.5*box[1];
        state[int(id_z+id_y*num_spacing+id_x*num_spacing*num_spacing)][2] = spacing_z*id_z-0.5*box[2];
        count += 1;
        id_z += 1;
        if(id_z==num_spacing):
            id_z = 0;
            id_y += 1;
        if(id_y==num_spacing):
            id_y = 0;
            id_x += 1;
    #Compute the pair distance
    dx = (state[0]-state[1])
    dx = dx-torch.round(dx/box[0])*box[0]
    
    #Re-compute one of the coordinates and shift to origin
    state[0] = dx/torch.norm(dx)*dist_init+state[1] 
    
    x_com = 0.5*(state[0]+state[1])
    for i in range(Np):
        state[i] -= x_com
        state[i] -= torch.round(state[i]/box[0])*box[0]
    return state;

start = CubicLattice(dist_init_start)
end = CubicLattice(dist_init_end)
initial_config = initializer(rank/(world_size-1))

committor = CommittorNetBP(num_nodes=200, boxsize=box[0], Np=32,rc=2.5,sigma=1.0).to('cpu')

#Initial Training Loss
initloss = nn.MSELoss()
initoptimizer = ParallelAdam(committor.parameters(), lr=1.0e-2)#,momentum=0.95, nesterov=True)
#initoptimizer = ParallelSGD(committor.parameters(), lr=1e-2,momentum=0.95, nesterov=True)

#from torchsummary import summary
running_loss = 0.0
tolerance = 1e-3

#Initial training try to fit the committor to the initial condition
tolerance = 1e-4
#batch_sizes = [64]
#for size in batch_sizes:
for i in tqdm.tqdm(range(10**5)):
    # zero the parameter gradients
    initoptimizer.zero_grad()
    
    # forward + backward + optimize
    q_vals = committor(initial_config.view(-1,Np))
    targets = torch.ones_like(q_vals)*rank/(dist.get_world_size()-1)
    cost = initloss(q_vals, targets)#,committor,config,cx)
    cost.backward()
    
    #Stepping up
    initoptimizer.step()
    with torch.no_grad():
        dist.all_reduce(cost)
        if i % 10 == 0 and rank == 0:
            print(i,cost.item() / world_size)#, committor(ftslayer.string[-1]))
        #    torch.save(committor.state_dict(), "initial_1hl_nn")#_{}".format(size))#prefix,rank+1))
            torch.save(committor.state_dict(), "initial_1hl_nn_bp")#_{}".format(size))#prefix,rank+1))
        if cost.item() / world_size < tolerance:
            if rank == 0:
                torch.save(committor.state_dict(), "initial_1hl_nn_bp")#_{}".format(size))#prefix,rank+1))
            print("Early Break!")
            break
    committor.zero_grad()
