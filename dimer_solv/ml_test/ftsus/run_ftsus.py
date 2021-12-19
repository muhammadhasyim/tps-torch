import sys
sys.path.append("..")

#Import necessarry tools from torch
import tpstorch
import torch
import torch.distributed as dist
import torch.nn as nn
import scipy.spatial

#Import necessarry tools from tpstorch 
from dimer_ftsus_nosolv import DimerFTSUS
from committor_nn import initializeConfig, CommittorNet, CommittorNetBP, CommittorNetDR
from dimer_ftsus_nosolv import FTSLayerUSCustom as FTSLayer
from tpstorch.ml.data import EXPReweightStringSimulation
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

#(1) Initialization
r0 = 2**(1/6.0)
width =  0.5*r0
Np = 30+2
box = [14.736125994561544, 14.736125994561544, 14.736125994561544]
kappa_perp = 400.0
kappa_par = 600.0
kT = 1.0

start, end, initial_config = initializeConfig(rank/(world_size-1), r0, width, box,Np)

#Initialize neural net
#committor = torch.jit.script(CommittorNetDR(num_nodes=2500, boxsize=box[0]).to('cpu'))
committor = torch.jit.script(CommittorNetBP(num_nodes=200, boxsize=box[0], Np=32,rc=2.5,sigma=1.0).to('cpu'))

#Initialize the string for FTS method
ftslayer = FTSLayer(react_config=start[:2].flatten(),prod_config=end[:2].flatten(),num_nodes=world_size,boxsize=box[0],kappa_perpend=kappa_perp, kappa_parallel=kappa_par, num_particles=2).to('cpu')
#Load the pre-initialized neural network and string
committor.load_state_dict(torch.load("../initial_1hl_nn_bp"))
ftslayer.load_state_dict(torch.load("../test_string_config"))

n_boundary_samples = 100
batch_size = 8
period = 25
dimer_sim_bc = DimerFTSUS(  param="param_bc",
                            config=initial_config.clone().detach(), 
                            rank=rank, 
                            beta=1/kT, 
                            kappa = 0.0, 
                            save_config=False, 
                            mpi_group = mpi_group, 
                            ftslayer=ftslayer,
                            output_time=batch_size*period
                            )
dimer_sim = DimerFTSUS( param="param",
                        config=initial_config.clone().detach(), 
                        rank=rank, 
                        beta=1/kT, 
                        kappa = kappa_perp, 
                        save_config=True, 
                        mpi_group = mpi_group, 
                        ftslayer=ftslayer,
                        output_time=batch_size*period
                        )

#Construct datarunner
datarunner = EXPReweightStringSimulation(dimer_sim, committor, period=period, batch_size=batch_size, dimN=Np*3)
#Construct optimizers
optimizer = ParallelAdam(committor.parameters(), lr=1e-3)

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
for epoch in range(1):
    if rank == 0:
        print("epoch: [{}]".format(epoch+1))
    for i in tqdm.tqdm(range(10000)):
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
            main_loss = loss.main_loss
            bc_loss = loss.bc_loss
            
            loss_io.write('{:d} {:.5E} {:.5E} \n'.format(i+1,main_loss.item(),bc_loss.item()))
            loss_io.flush()
            #Print statistics 
            if rank == 0:
                torch.save(committor.state_dict(), "{}_params_t_{}_{}".format(prefix,i,rank))
