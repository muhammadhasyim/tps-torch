import sys
sys.path.append("..")

#Import necessarry tools from torch
import tpstorch
import torch
import torch.distributed as dist
import torch.nn as nn

#Import necessarry tools from tpstorch 
from dimer_us import DimerUS
from committor_nn import CommittorNetBP, CommittorNetDR
from tpstorch.ml.data import EXPReweightSimulation
from tpstorch.ml.optim import ParallelAdam
from tpstorch.ml.nn import BKELossEXP
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

Np = 30+2
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

#Initialize neural net
#committor = CommittorNetBP(num_nodes=200, boxsize=box[0], Np=32, rc = 2.5, sigma=1.0).to('cpu')
#committor = torch.jit.script(CommittorNetDR(num_nodes=2500, boxsize=box[0]).to('cpu'))
committor = torch.jit.script(CommittorNetBP(num_nodes=200, boxsize=box[0], Np=32, rc = 2.5, sigma=1.0).to('cpu'))

kappa= 700#10
#Initialize the string for FTS method
#Load the pre-initialized neural network and string
#committor.load_state_dict(torch.load("../initial_1hl_nn"))
committor.load_state_dict(torch.load("../initial_1hl_nn_bp"))
kT = 1.0

n_boundary_samples = 100
batch_size = 8
period = 5
dimer_sim_bc = DimerUS(param="param_bc",config=initial_config.clone().detach(), rank=rank, beta=1/kT, kappa = 0.0, save_config=False, mpi_group = mpi_group, output_time=batch_size*period)
dimer_sim = DimerUS(param="param",config=initial_config.clone().detach(), rank=rank, beta=1/kT, kappa = kappa, save_config=True, mpi_group = mpi_group, output_time=batch_size*period)
dimer_sim_com = DimerUS(param="param",config=initial_config.clone().detach(), rank=rank, beta=1/kT, kappa = 0.0,save_config=True, mpi_group = mpi_group, output_time=batch_size*period)

#Construct FTSSimulation
datarunner = EXPReweightSimulation(dimer_sim, committor, period=period, batch_size=batch_size, dimN=Np*3)

#Initialize main loss function and optimizers
#Optimizer, doing EXP Reweighting. We can do SGD (integral control), or Heavy-Ball (PID control)
loss = BKELossEXP(  bc_sampler = dimer_sim_bc,
                    committor = committor,
                    lambda_A = 1e4,
                    lambda_B = 1e4,
                    start_react = start,
                    start_prod = end,
                    n_bc_samples = 200, 
                    bc_period = 10,
                    batch_size_bc = 0.5,
                    )

loss_io = []
loss_io = open("{}_statistic_{}.txt".format(prefix,rank+1),'w')

#Training loop
optimizer = ParallelAdam(committor.parameters(), lr=1.0e-3)

#We can train in terms of epochs, but we will keep it in one epoch
for epoch in range(1):
    if rank == 0:
        print("epoch: [{}]".format(epoch+1))
    for i in tqdm.tqdm(range(10000)):#20000)):
        # get data and reweighting factors
        config, grad_xs, invc, fwd_wl, bwrd_wl = datarunner.runSimulation()
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # (2) Update the neural network
        # forward + backward + optimize
        cost = loss(grad_xs,invc,fwd_wl,bwrd_wl)
        cost.backward()
        
        optimizer.step()

        # print statistics
        with torch.no_grad():
            #if counter % 10 == 0:
            main_loss = loss.main_loss
            bc_loss = loss.bc_loss
            
            loss_io.write('{:d} {:.5E} {:.5E} {:.5E} \n'.format(i+1,main_loss.item(),bc_loss.item()))#,cm_loss.item()))
            loss_io.flush()
            #Print statistics 
            if rank == 0:
                torch.save(committor.state_dict(), "{}_params_t_{}_{}".format(prefix,i,rank))
