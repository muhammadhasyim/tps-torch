#import sys
#sys.path.insert(0,"/global/home/users/muhammad_hasyim/tps-torch/build")

#Import necessarry tools from torch
import tpstorch
import torch
import torch.distributed as dist
import torch.nn as nn
import scipy.spatial

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

Np = 30+2

#This function only assumes that the string consists of the dimer without solvent particles
@torch.no_grad()
def dimer_nullspace(vec,x,boxsize):
    Np = 32
    ##(1) Pre-processing so that dimer is at the center
    old_x = x.view(Np,3).clone()

    #Compute the pair distance
    dx = (old_x[0]-old_x[1])
    dx = dx-torch.round(dx/boxsize)*boxsize
    
    #Re-compute one of the coordinates and shift to origin
    old_x[0] = dx+old_x[1] 
    x_com = 0.5*(old_x[0]+old_x[1])
    
    new_vec = vec.view(Np,3).clone()
    #Compute the pair distance
    ds = (new_vec[0]-new_vec[1])
    ds = ds-torch.round(ds/boxsize)*boxsize
      
    #Re-compute one of the coordinates and shift to origin
    new_vec[0] = ds+new_vec[1]
    s_com = 0.5*(new_vec[0]+new_vec[1])
    for i in range(Np):
        old_x[i] -= x_com
        new_vec[i] -= s_com
        old_x[i] -= torch.round(old_x[i]/boxsize)*boxsize 
        new_vec[i] -= torch.round(new_vec[i]/boxsize)*boxsize 
   
    ##(2) Rotate the system using Kabsch algorithm
    #weights = np.ones(Np)
    #weights = np.zeros(Np)
    #weights[0] = 1.0
    #weights[1] = 1.0
    weights = np.ones(Np)#/(Np-2)
    #weights = np.zeros(self.Np)#np.ones(self.Np)/(self.Np-2)
    weights[0] = 1.0
    weights[1] = 1.0
    rotate,rmsd = scipy.spatial.transform.Rotation.align_vectors(new_vec.numpy(),old_x.numpy(), weights=weights)
    for i in range(Np):
        old_x[i] = torch.tensor(rotate.apply(old_x[i].numpy())) 
        old_x[i] -= torch.round(old_x[i]/boxsize)*boxsize 
    return old_x.flatten()

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

#Initialize the string
#For now I'm only going to interpolate through the dimer, ignoring solvent particles
ftslayer = FTSLayer(react_config=start.flatten(),prod_config=end.flatten(),num_nodes=world_size,boxsize=box[0],num_particles=Np).to('cpu')

#Construct FTSSimulation
ftsoptimizer = FTSUpdate(ftslayer.parameters(), deltatau=0.01,kappa=0.1,periodic=True,dim=3)
kT = 1.0
batch_size = 10
period = 10
#Initialize the dimer simulation
#dimer_sim_fts = DimerFTS(param="param",config=initial_config.detach().clone(), rank=rank, beta=1/kT, save_config=True, mpi_group = mpi_group, ftslayer=ftslayer,output_time=batch_size*period)
dimer_sim_fts = DimerFTS(param="param",config=ftslayer.string[rank].view(Np,3).detach().clone(), rank=rank, beta=1/kT, save_config=True, mpi_group = mpi_group, ftslayer=ftslayer,output_time=batch_size*period)
datarunner_fts = FTSSimulation(dimer_sim_fts, nn_training = False, period=period, batch_size=batch_size, dimN=Np*3)

#FTS Simulation Training Loop
with open("string_{}_config.xyz".format(rank),"w") as f, open("string_{}_log.txt".format(rank),"w") as g:
    for i in tqdm.tqdm(range(10000)):
        # get data and reweighting factors
        configs = datarunner_fts.runSimulation()
        ftsoptimizer.step(configs,len(configs),boxsize=box[0],remove_nullspace=dimer_nullspace)
        string_temp = ftslayer.string[rank].view(Np,3)
        
        f.write("32 \n")
        f.write('Lattice=\"{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}\" '.format(box[0],box[1],box[2]))
        f.write('Origin=\"{} {} {}\" '.format(-0.5*box[0],-0.5*box[1],-0.5*box[2]))
        f.write("Properties=type:S:1:pos:R:3:aux1:R:1 \n")
        f.write("B {} {} {} {} \n".format(string_temp[0,0],string_temp[0,1], string_temp[0,2],0.5*r0))
        f.write("B {} {} {} {} \n".format(string_temp[1,0],string_temp[1,1], string_temp[1,2],0.5*r0))
        for i in range(2,Np):
            f.write("A {} {} {} {} \n".format(string_temp[i,0],string_temp[i,1], string_temp[i,2],0.5*r0))
        f.flush()
        g.write("{} {} \n".format((i+1)*period,torch.norm(string_temp[0]-string_temp[1])))
        g.flush()
        if rank == 0:
            torch.save(ftslayer.state_dict(), "test_string_config")
