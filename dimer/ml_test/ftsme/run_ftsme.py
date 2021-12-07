import sys
sys.path.append("..")

#Import necessarry tools from torch
import tpstorch
import torch
import torch.distributed as dist
import torch.nn as nn

#Import necessarry tools from tpstorch 
from dimer_fts import DimerFTS
from committor_nn import CommittorNet, CommittorNetDR
from dimer_fts import FTSLayerCustom as FTSLayer
from tpstorch.ml.data import FTSSimulation#, EXPReweightStringSimulation
from tpstorch.ml.optim import ParallelSGD, ParallelAdam, FTSUpdate
#from tpstorch.ml.nn import BKELossEXP
from tpstorch.ml.nn import BKELossFTS
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

@torch.no_grad()
def dimer_nullspace(vec,x,boxsize):
    ##(1) Remove center of mass 
    old_x = x.view(2,3).clone()

    #Compute the pair distance
    dx = (old_x[0]-old_x[1])
    dx = dx-torch.round(dx/boxsize)*boxsize
    
    #Re-compute one of the coordinates and shift to origin
    old_x[0] = dx+old_x[1] 
    x_com = 0.5*(old_x[0]+old_x[1])
    old_x[0] -= x_com
    old_x[1] -= x_com
    
    vec = vec.view(2,3).clone()
    
    #Compute the pair distance
    ds = (vec[0]-vec[1])
    ds = ds-torch.round(ds/boxsize)*boxsize
    
    #Re-compute one of the coordinates and shift to origin
    vec[0] = ds+vec[1]
    s_com = 0.5*(vec[0]+vec[1])#.detach().clone()#,dim=1)
    vec[0] -= s_com
    vec[1] -= s_com
        
    ##(2) Rotate the configuration
    dx /= torch.norm(dx)
    ds /= torch.norm(ds)
    new_x = torch.zeros_like(old_x)    
    v = torch.cross(dx,ds)
    cosine = torch.dot(ds,dx)
    new_x[0] = old_x[0] +torch.cross(v,old_x[0])+torch.cross(v,torch.cross(v,old_x[0]))/(1+cosine)
    new_x[1] = old_x[1] +torch.cross(v,old_x[1])+torch.cross(v,torch.cross(v,old_x[1]))/(1+cosine)
    return new_x.flatten()


#Initialize neural net
#Initialization
r0 = 2**(1/6.0)
width =  0.5*r0

#Reactant
dist_init = r0-0.95*r0
start = torch.zeros((2,3))
start[0][2] = -0.5*dist_init
start[1][2] = 0.5*dist_init

#Product state
dist_init = r0+2*width+0.95*r0
end = torch.zeros((2,3))
end[0][2] = -0.5*dist_init
end[1][2] = 0.5*dist_init


#Initialize neural net
#committor = CommittorNet(d=6,num_nodes=2500).to('cpu')
committor = CommittorNetDR(num_nodes=2500, boxsize=10).to('cpu')

#Initialize the string for FTS method
ftslayer = FTSLayer(react_config=start.flatten(),prod_config=end.flatten(),num_nodes=world_size,boxsize=10.0).to('cpu')

#Load the pre-initialized neural network and string
committor.load_state_dict(torch.load("../initial_1hl_nn"))
kT = 1.0
ftslayer.load_state_dict(torch.load("../test_string_config"))

n_boundary_samples = 100
batch_size = 8
period = 25
dimer_sim_bc = DimerFTS(param="param_bc",config=ftslayer.string[rank].view(2,3).clone().detach(), rank=rank, beta=1/kT, save_config=False, mpi_group = mpi_group, ftslayer=ftslayer,output_time=batch_size*period)
dimer_sim = DimerFTS(param="param",config=ftslayer.string[rank].view(2,3).clone().detach(), rank=rank, beta=1/kT, save_config=True, mpi_group = mpi_group, ftslayer=ftslayer,output_time=batch_size*period)

#Construct FTSSimulation
ftsoptimizer = FTSUpdate(ftslayer.parameters(), deltatau=0.02,momentum=0.9,nesterov=True,kappa=0.1,periodic=True,dim=3)
datarunner = FTSSimulation(dimer_sim, committor = committor, nn_training = True, period=period, batch_size=batch_size, dimN=6)

#Initialize main loss function and optimizers
#Optimizer, doing EXP Reweighting. We can do SGD (integral control), or Heavy-Ball (PID control)
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

loss_io = []
loss_io = open("{}_statistic_{}.txt".format(prefix,rank+1),'w')

#Training loop
optimizer = ParallelAdam(committor.parameters(), lr=3e-3)

#We can train in terms of epochs, but we will keep it in one epoch
with open("string_{}_config.xyz".format(rank),"w") as f, open("string_{}_log.txt".format(rank),"w") as g:
    for epoch in range(1):
        if rank == 0:
            print("epoch: [{}]".format(epoch+1))
        for i in tqdm.tqdm(range(1000)):#20000)):
            # get data and reweighting factors
            configs, grad_xs  = datarunner.runSimulation()
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # (2) Update the neural network
            # forward + backward + optimize
            cost = loss(grad_xs, dimer_sim.rejection_count)
            cost.backward()
            
            optimizer.step()
            
            ftsoptimizer.step(configs,len(configs),boxsize=10.0,remove_nullspace=dimer_nullspace)

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
