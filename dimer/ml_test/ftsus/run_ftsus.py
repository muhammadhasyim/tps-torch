import sys
sys.path.append("..")

#Import necessarry tools from torch
import tpstorch
import torch
import torch.distributed as dist
import torch.nn as nn

#Import necessarry tools from tpstorch 
from dimer_ftsus import DimerFTSUS
from committor_nn import CommittorNet, CommittorNetDR
from dimer_ftsus import FTSLayerUSCustom as FTSLayer
from tpstorch.ml.data import FTSSimulation, EXPReweightStringSimulation
from tpstorch.ml.optim import ParallelSGD, ParallelAdam
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

#Initialize neural net
#Initialization
r0 = 2**(1/6.0)
width =  0.5*r0

#Reactant
dist_init = r0
start = torch.zeros((2,3))
start[0][2] = -0.5*dist_init
start[1][2] = 0.5*dist_init

#Product state
dist_init = r0+2*width
end = torch.zeros((2,3))
end[0][2] = -0.5*dist_init
end[1][2] = 0.5*dist_init


#Initialize neural net
#committor = CommittorNet(d=6,num_nodes=2500).to('cpu')
committor = CommittorNetDR(num_nodes=2500, boxsize=10).to('cpu')

kappa_perp = 200.0#60#10
kappa_par = 500.0#60
#Initialize the string for FTS method
ftslayer = FTSLayer(react_config=start.flatten(),prod_config=end.flatten(),num_nodes=world_size,boxsize=10.0,kappa_perpend=kappa_perp,kappa_parallel=kappa_par).to('cpu')

#Load the pre-initialized neural network and string
committor.load_state_dict(torch.load("../initial_1hl_ftsus_nn"))
kT = 1.0
ftslayer.load_state_dict(torch.load("../test_string_config"))

n_boundary_samples = 100
batch_size = 8
period = 25
dimer_sim_bc = DimerFTSUS(param="param_bc",config=ftslayer.string[rank].view(2,3).clone().detach(), rank=rank, beta=1/kT, kappa = 0.0, save_config=False, mpi_group = mpi_group, ftslayer=ftslayer,output_time=batch_size*period)
dimer_sim = DimerFTSUS(param="param",config=ftslayer.string[rank].view(2,3).clone().detach(), rank=rank, beta=1/kT, kappa = kappa_perp, save_config=True, mpi_group = mpi_group, ftslayer=ftslayer,output_time=batch_size*period)

#Construct FTSSimulation
datarunner = EXPReweightStringSimulation(dimer_sim, committor, period=period, batch_size=batch_size, dimN=6)

#Initialize main loss function and optimizers
#Optimizer, doing EXP Reweighting. We can do SGD (integral control), or Heavy-Ball (PID control)
loss = BKELossEXP(  bc_sampler = dimer_sim_bc,
                    committor = committor,
                    lambda_A = 1e4,
                    lambda_B = 1e4,
                    start_react = start,
                    start_prod = end,
                    n_bc_samples = 100, 
                    bc_period = 10,
                    batch_size_bc = 0.5,
                    )

loss_io = []
loss_io = open("{}_statistic_{}.txt".format(prefix,rank+1),'w')

#Training loop
optimizer = ParallelAdam(committor.parameters(), lr=1e-3)

#We can train in terms of epochs, but we will keep it in one epoch
for epoch in range(1):
    if rank == 0:
        print("epoch: [{}]".format(epoch+1))
    for i in tqdm.tqdm(range(1000)):
        # get data and reweighting factors
        config, grad_xs, invc, fwd_wl, bwrd_wl = datarunner.runSimulation()
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # (2) Update the neural network
        # forward + backward + optimize
        cost = loss(grad_xs,invc,fwd_wl,bwrd_wl)
        cost.backward()#retain_graph=True)
        
        optimizer.step()

        # print statistics
        with torch.no_grad():
            #if counter % 10 == 0:
            main_loss = loss.main_loss
            bc_loss = loss.bc_loss
            
            loss_io.write('{:d} {:.5E} {:.5E} \n'.format(i+1,main_loss.item(),bc_loss.item()))
            loss_io.flush()
            #Print statistics 
            if rank == 0:
                torch.save(committor.state_dict(), "{}_params_t_{}_{}".format(prefix,i,rank))
