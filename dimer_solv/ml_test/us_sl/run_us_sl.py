import sys
sys.path.append("..")

#Import necessarry tools from torch
import tpstorch
import torch
import torch.distributed as dist
import torch.nn as nn

#Import necessarry tools from tpstorch 
from dimer_us import DimerUS
from committor_nn import initializeConfig, CommittorNetBP, CommittorNetDR
from tpstorch.ml.data import EXPReweightSimulation
from tpstorch.ml.optim import ParallelAdam
from tpstorch.ml.nn import BKELossEXP, CommittorLoss2
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
kappa= 700
kT = 1.0

start, end, initial_config = initializeConfig(rank/(world_size-1), r0, width, box,Np)

#Initialize neural net
#committor = torch.jit.script(CommittorNetDR(num_nodes=2500, boxsize=box[0]).to('cpu'))
committor = torch.jit.script(CommittorNetBP(num_nodes=200, boxsize=box[0], Np=32, rc = 2.5, sigma=1.0).to('cpu'))
committor.load_state_dict(torch.load("../initial_1hl_nn_bp"))

n_boundary_samples = 100
batch_size = 8
period = 25
dimer_sim_bc = DimerUS( param="param_bc",
                        config=initial_config.clone().detach(), 
                        rank=rank, 
                        beta=1/kT, 
                        kappa = 0.0, 
                        save_config=False, 
                        mpi_group = mpi_group, 
                        output_time=batch_size*period
                        )
dimer_sim = DimerUS(    param="param",
                        config=initial_config.clone().detach(), 
                        rank=rank, 
                        beta=1/kT, 
                        kappa = kappa, 
                        save_config=True, 
                        mpi_group = mpi_group, 
                        output_time=batch_size*period
                        )
dimer_sim_com = DimerUS(param="param",
                        config=initial_config.clone().detach(), 
                        rank=rank, 
                        beta=1/kT, 
                        kappa = 0.0, 
                        save_config=False, 
                        mpi_group = mpi_group, 
                        output_time=batch_size*period
                        )

#Construct datarunner
datarunner = EXPReweightSimulation(dimer_sim, committor, period=period, batch_size=batch_size, dimN=Np*3)

#The BKE Loss function, with EXP Reweighting
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

#The supervised learning loss
cmloss = CommittorLoss2( cl_sampler = dimer_sim_com,
                        committor = committor,
                        lambda_cl=100.0,
                        cl_start=10,
                        cl_end=200,
                        cl_rate=10,
                        cl_trials=40,
                        batch_size_cl=0.5
                        )

#Some setup to increase penalty strength
lambda_cl_end = 10**3
cl_start=200
cl_end=10000
cl_stepsize = (lambda_cl_end-cmloss.lambda_cl)/(cl_end-cl_start)

#Initialize main loss function and optimizers
#Optimizer, doing EXP Reweighting. We can do SGD (integral control), or Heavy-Ball (PID control)
optimizer = ParallelAdam(committor.parameters(), lr=1.0e-3)

loss_io = []
loss_io = open("{}_statistic_{}.txt".format(prefix,rank+1),'w')

#Training loop
for epoch in range(1):
    if rank == 0:
        print("epoch: [{}]".format(epoch+1))
    for i in tqdm.tqdm(range(10000)):
        if (i > cl_start) and (i <= cl_end):
            cmloss.lambda_cl += cl_stepsize
        elif i > cl_end:
            cmloss.lambda_cl = lambda_cl_end
        # get data and reweighting factors
        config, grad_xs, invc, fwd_wl, bwrd_wl = datarunner.runSimulation()
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # (2) Update the neural network
        # forward + backward + optimize
        bkecost = loss(grad_xs,invc,fwd_wl,bwrd_wl)
        cmcost = cmloss(i, dimer_sim.getConfig())
        cost = bkecost+cmcost
        cost.backward()
        
        optimizer.step()

        # print statistics
        with torch.no_grad():
            main_loss = loss.main_loss
            cm_loss = cmloss.cl_loss
            bc_loss = loss.bc_loss
            
            loss_io.write('{:d} {:.5E} {:.5E} {:.5E} \n'.format(i+1,main_loss.item(),bc_loss.item(),cm_loss.item()))
            loss_io.flush()
            #Print statistics 
            if rank == 0:
                torch.save(committor.state_dict(), "{}_params_t_{}_{}".format(prefix,i,rank))
