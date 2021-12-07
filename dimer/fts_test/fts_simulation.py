#import sys
#sys.path.insert(0,"/global/home/users/muhammad_hasyim/tps-torch/build")

#Import necessarry tools from torch
import tpstorch
import torch
import torch.distributed as dist
import torch.nn as nn

#Import necessarry tools from tpstorch 
#from mb_fts import MullerBrown as MullerBrownFTS#, CommittorNet
from dimer_fts import DimerFTS
from dimer_fts import FTSLayerCustom as FTSLayer
#from tpstorch.ml.data import FTSSimulation#, EXPReweightStringSimulation
from tpstorch.ml.data import FTSSimulation#, EXPReweightStringSimulation
from tpstorch.ml.optim import ParallelSGD, ParallelAdam, FTSImplicitUpdate, FTSUpdate
#from tpstorch.ml.nn import FTSLayer#BKELossFTS, BKELossEXP, FTSCommittorLoss, FTSLayer

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

@torch.no_grad()
def reset_orientation(vec,boxsize):
    #Remove center of mass 
    vec = vec.view(2,3).clone()
    s_com = (0.5*(vec[0]+vec[1])).detach().clone()#,dim=1)
    vec[0] -= s_com
    vec[1] -= s_com
    #Create the orientation vector
    ds = (vec[0]-vec[1])#/torch.norm(vec[0]-vec[1])
    ds = ds-torch.round(ds/boxsize)*boxsize
    ds /= torch.norm(ds)
    
    #We want to remove center of mass in x and string
    x = torch.zeros((2,3))
    x[0,2] = -1.0
    x[1,2] = 1.0
    dx = (x[0]-x[1])
    dx /= torch.norm(x[0]-x[1])
        
    #Rotate the configuration
    v = torch.cross(ds,dx)
    cosine = torch.dot(ds,dx)
    vec[0] += torch.cross(v,vec[0])+torch.cross(v,torch.cross(v,vec[0]))/(1+cosine)
    #vec[0] -= torch.round(vec[0]/boxsize)*boxsize
    vec[1] += torch.cross(v,vec[1])+torch.cross(v,torch.cross(v,vec[1]))/(1+cosine)
    #vec[1] -= torch.round(vec[1]/boxsize)*boxsize
    return vec.flatten()

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

#Initialize the string
ftslayer = FTSLayer(react_config=start.flatten(),prod_config=end.flatten(),num_nodes=world_size,boxsize=10.0).to('cpu')


#Construct FTSSimulation
ftsoptimizer = FTSUpdate(ftslayer.parameters(), deltatau=0.05,momentum=0.9,nesterov=True, kappa=0.1,periodic=True,dim=3)
kT = 1.0
batch_size = 10
period = 10
#Initialize the dimer
dimer_sim_fts = DimerFTS(param="param",config=ftslayer.string[rank].view(2,3).detach().clone(), rank=rank, beta=1/kT, save_config=True, mpi_group = mpi_group, ftslayer=ftslayer,output_time=batch_size*period)
datarunner_fts = FTSSimulation(dimer_sim_fts, nn_training = False, period=period, batch_size=batch_size, dimN=6)

#FTS Simulation Training Loop
with open("string_{}_config.xyz".format(rank),"w") as f, open("string_{}_log.txt".format(rank),"w") as g:
    for i in tqdm.tqdm(range(500)):
        # get data and reweighting factors
        configs = datarunner_fts.runSimulation()
        ftsoptimizer.step(configs,len(configs),boxsize=10.0,remove_nullspace=dimer_nullspace,reset_orient=reset_orientation)
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
        if rank == 0:
            torch.save(ftslayer.state_dict(), "test_string_config")
