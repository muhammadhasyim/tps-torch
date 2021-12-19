import sys
sys.path.append("..")

#Import necessarry tools from torch
import tpstorch
import torch
import torch.distributed as dist
import torch.nn as nn
import scipy.spatial

#Import necessarry tools from tpstorch 
from dimer_fts_nosolv import dimer_reorient, DimerFTS
from committor_nn import initializeConfig, CommittorNet, CommittorNetBP, CommittorNetDR
from dimer_fts_nosolv import FTSLayerCustom as FTSLayer
from tpstorch.ml.data import FTSSimulation
from tpstorch.ml.optim import ParallelSGD, ParallelAdam, FTSUpdate
from tpstorch.ml.nn import BKELossFTS, CommittorLoss2
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
kT = 1.0

start, end, initial_config = initializeConfig(rank/(world_size-1), r0, width, box,Np)

#Initialize neural net
#committor = CommittorNetDR(num_nodes=2500, boxsize=10).to('cpu')
committor = torch.jit.script(CommittorNetBP(num_nodes=200, boxsize=box[0], Np=32,rc=2.5,sigma=1.0).to('cpu'))

#Initialize the string for FTS method
ftslayer = FTSLayer(react_config=start[:2].flatten(),prod_config=end[:2].flatten(),num_nodes=world_size,boxsize=box[0],num_particles=2).to('cpu')

#Load the pre-initialized neural network and string
committor.load_state_dict(torch.load("../initial_1hl_nn_bp"))
ftslayer.load_state_dict(torch.load("../test_string_config"))

n_boundary_samples = 100
batch_size = 8
period = 25
#Initialize the dimer simulation
#Initialize the dimer simulation
dimer_sim_bc = DimerFTS(param="param_bc",
                        config=initial_config.detach().clone(), 
                        rank=rank, 
                        beta=1/kT, 
                        save_config=False, 
                        mpi_group = mpi_group, 
                        ftslayer=ftslayer,
                        output_time=batch_size*period
                        )
dimer_sim = DimerFTS(   param="param",
                        config=initial_config.detach().clone(), 
                        rank=rank, 
                        beta=1/kT, 
                        save_config=True, 
                        mpi_group = mpi_group, 
                        ftslayer=ftslayer,
                        output_time=batch_size*period
                        )
dimer_sim_com = DimerFTS(param="param",
                        config=initial_config.detach().clone(), 
                        rank=rank, 
                        beta=1/kT, 
                        save_config=False, 
                        mpi_group = mpi_group, 
                        ftslayer=ftslayer,
                        output_time=batch_size*period
                        )

#Construct datarunner
datarunner = FTSSimulation(dimer_sim, committor = committor, nn_training = True, period=period, batch_size=batch_size, dimN=Np*3)
#Construct optimizers
ftsoptimizer = FTSUpdate(ftslayer.parameters(), deltatau=0.01,momentum=0.9,nesterov=True,kappa=0.1,periodic=True,dim=3)
optimizer = ParallelAdam(committor.parameters(), lr=3e-3)

#Initialize loss functions
loss = BKELossFTS(  bc_sampler = dimer_sim_bc,
                    committor = committor,
                    lambda_A = 1e4,
                    lambda_B = 1e4,
                    start_react = start,
                    start_prod = end,
                    n_bc_samples = 100, 
                    bc_period = 10,
                    batch_size_bc = 0.5,
                    tol = 5e-10,
                    mode= 'shift')

cmloss = CommittorLoss2( cl_sampler = dimer_sim_com,
                        committor = committor,
                        lambda_cl=100.0,
                        cl_start=10,
                        cl_end=200,
                        cl_rate=10,
                        cl_trials=50,
                        batch_size_cl=0.5
                        )

loss_io = []
loss_io = open("{}_statistic_{}.txt".format(prefix,rank+1),'w')

#Training loop
lambda_cl_end = 10**3
cl_start=200
cl_end=10000
cl_stepsize = (lambda_cl_end-cmloss.lambda_cl)/(cl_end-cl_start)

#We can train in terms of epochs, but we will keep it in one epoch
with open("string_{}_config.xyz".format(rank),"w") as f, open("string_{}_log.txt".format(rank),"w") as g:
    for epoch in range(1):
        if rank == 0:
            print("epoch: [{}]".format(epoch+1))
        for i in tqdm.tqdm(range(10000)):#20000)):
            if (i > cl_start) and (i <= cl_end):
                cmloss.lambda_cl += cl_stepsize
            elif i > cl_end:
                cmloss.lambda_cl = lambda_cl_end
            # get data and reweighting factors
            configs, grad_xs  = datarunner.runSimulation()
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # (2) Update the neural network
            # forward + backward + optimize
            bkecost = loss(grad_xs, dimer_sim.rejection_count)
            cmcost = cmloss(i, dimer_sim.getConfig())
            cost = bkecost+cmcost
            cost.backward()
            
            optimizer.step()
            
            ftsoptimizer.step(configs[:,:6],len(configs),boxsize=box[0],reorient_sample=dimer_reorient)

            # print statistics
            with torch.no_grad():
                string_temp = ftslayer.string[rank].view(2,3)
                f.write("2 \n")
                f.write('Lattice=\"10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0\" ')
                f.write('Origin=\"-5.0 -5.0 -5.0\" ')
                f.write("Properties=type:S:1:pos:R:3:aux1:R:1 \n")
                f.write("2 {} {} {} {} \n".format(string_temp[0,0],string_temp[0,1], string_temp[0,2],0.5*r0))
                f.write("2 {} {} {} {} \n".format(string_temp[1,0],string_temp[1,1], string_temp[1,2],0.5*r0))
                f.flush()
                g.write("{} {} \n".format((i+1)*period,torch.norm(string_temp[0]-string_temp[1])))
                g.flush()
                #if counter % 10 == 0:
                main_loss = loss.main_loss
                bc_loss = loss.bc_loss
                
                loss_io.write('{:d} {:.5E} {:.5E} \n'.format(i+1,main_loss.item(),bc_loss.item()))
                loss_io.flush()
                #Print statistics 
                if rank == 0:
                    torch.save(committor.state_dict(), "{}_params_t_{}_{}".format(prefix,i,rank))
