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
    x = x.view(2,3)
    x_com = 0.5*(x[0]+x[1])
    x[0] -= x_com
    x[1] -= x_com
    dx = (x[0]-x[1])
    dx = dx-torch.round(dx/boxsize)*boxsize
    dx /= torch.norm(x[0]-x[1])
        
    #Rotate the configuration
    new_x = torch.zeros_like(x)    
    v = torch.cross(dx,ds)
    cosine = torch.dot(ds,dx)
    new_x[0] = x[0] +torch.cross(v,x[0])+torch.cross(v,torch.cross(v,x[0]))/(1+cosine)
    new_x[0] = new_x[0]-torch.round(new_x[0]/boxsize)*boxsize
    new_x[1] = x[1] +torch.cross(v,x[1])+torch.cross(v,torch.cross(v,x[1]))/(1+cosine)
    new_x[1] = new_x[1]-torch.round(new_x[1]/boxsize)*boxsize
    return new_x.flatten()

#Initialization
r0 = 2**(1/6.0)
width =  0.5*r0

#Reactant
dist_init = r0-0.25*r0
start = torch.zeros((2,3))
start[0][2] = -0.5*dist_init
start[1][2] = 0.5*dist_init

#Product state
dist_init = r0+2*width+0.25*r0
end = torch.zeros((2,3))
end[0][2] = -0.5*dist_init
end[1][2] = 0.5*dist_init

#Initialize the string
ftslayer = FTSLayer(react_config=start.flatten(),prod_config=end.flatten(),num_nodes=world_size,boxsize=10.0).to('cpu')#,kappa_perpend=0.0,kappa_parallel=0.0).to('cpu')

#Initialize the dimer
kT = 1.0#1.0
dimer_sim_fts = DimerFTS(param="param",config=ftslayer.string[rank].view(2,3).detach().clone(), rank=rank, beta=1/kT, save_config=True, mpi_group = mpi_group, ftslayer=ftslayer)

#Construct FTSSimulation
#ftsoptimizer = FTSImplicitUpdate(ftslayer.parameters(), dimN = 6, deltatau=0.005,kappa=0.2,periodic=True,dim=3)
ftsoptimizer = FTSUpdate(ftslayer.parameters(), deltatau=0.05,momentum=0.95,kappa=0.1,periodic=True,dim=3)
batch_size = 10
period = 10
datarunner_fts = FTSSimulation(dimer_sim_fts, nn_training = False, period=period, batch_size=batch_size, dimN=6)

#FTS Simulation Training Loop
with open("string_{}_config.xyz".format(rank),"w") as f, open("string_{}_log.txt".format(rank),"w") as g:
    for i in tqdm.tqdm(range(300)):
        # get data and reweighting factors
        configs = datarunner_fts.runSimulation()
        ftsoptimizer.step(configs,len(configs),boxsize=10.0,remove_nullspace=dimer_nullspace)#,reset_orient=reset_orientation)
        string_temp = ftslayer.string[rank].view(2,3)
        #print(torch.norm(string_temp[0][0]-string_temp[0][1]),r0,torch.norm(string_temp[-1][0]-string_temp[-1][1]),r0+2*width)
        f.write("2 \n")
        f.write("#FTS step {} \n".format(i+1))
        f.write("1 {} {} {} {} \n".format(0.5*r0,string_temp[0,0],string_temp[0,1], string_temp[0,2]))##String config number "+rank+"\n")
        f.write("1 {} {} {} {} \n".format(0.5*r0,string_temp[1,0],string_temp[1,1], string_temp[1,2]))##String config number "+rank+"\n")
        f.flush()
        g.write("{} {}".format((i+1)*period,torch.norm(string_temp[0]-string_temp[1])))
        if rank == 0:
            torch.save(ftslayer.state_dict(), "final_{}".format(i))
