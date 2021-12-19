import sys
sys.path.append("..")

#Import necessarry tools from torch
import tpstorch
import torch
import torch.distributed as dist
import torch.nn as nn
import scipy.spatial

#Import necessarry tools from tpstorch 
from dimer_fts_nosolv import DimerFTS
from committor_nn import CommittorNet, CommittorNetBP, CommittorNetDR
from dimer_fts_nosolv import FTSLayerCustom as FTSLayer
from tpstorch.ml.data import FTSSimulation#, EXPReweightStringSimulation
from tpstorch.ml.optim import ParallelSGD, ParallelAdam, FTSUpdate
#from tpstorch.ml.nn import BKELossEXP
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
Np = 32
def initializer(s):
    return (1-s)*start+s*end
#This function only assumes that the string consists of the dimer without solvent particles
@torch.no_grad()
def dimer_nullspace(vec,x,boxsize):
    Np = 2
    ##(1) Pre-processing so that dimer is at the center
    old_x = x.view(2,3).clone()

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
    #weights = np.ones(Np)/(Np-2)
    weights = np.zeros(Np)#np.ones(self.Np)/(self.Np-2)
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


#Initialize neural net
#committor = CommittorNetDR(num_nodes=2500, boxsize=10).to('cpu')
committor = torch.jit.script(CommittorNetBP(num_nodes=200, boxsize=box[0], Np=32,rc=2.5,sigma=1.0).to('cpu'))

#Initialize the string for FTS method
ftslayer = FTSLayer(react_config=start[:2].flatten(),prod_config=end[:2].flatten(),num_nodes=world_size,boxsize=box[0],num_particles=2).to('cpu')

#Load the pre-initialized neural network and string
committor.load_state_dict(torch.load("../initial_1hl_nn_bp"))
kT = 1.0
ftslayer.load_state_dict(torch.load("../test_string_config"))

n_boundary_samples = 100
batch_size = 8
period = 25
#Initialize the dimer simulation
dimer_sim_bc = DimerFTS(param="param_bc",config=initial_config.detach().clone(), rank=rank, beta=1/kT, save_config=False, mpi_group = mpi_group, ftslayer=ftslayer,output_time=batch_size*period)
dimer_sim = DimerFTS(param="param",config=initial_config.detach().clone(), rank=rank, beta=1/kT, save_config=True, mpi_group = mpi_group, ftslayer=ftslayer,output_time=batch_size*period)
dimer_sim_com = DimerFTS(param="param",config=initial_config.detach().clone(), rank=rank, beta=1/kT, save_config=False, mpi_group = mpi_group, ftslayer=ftslayer,output_time=batch_size*period)

#Construct FTSSimulation
ftsoptimizer = FTSUpdate(ftslayer.parameters(), deltatau=0.02,momentum=0.9,nesterov=True,kappa=0.1,periodic=True,dim=3)
datarunner = FTSSimulation(dimer_sim, committor = committor, nn_training = True, period=period, batch_size=batch_size, dimN=Np*3)

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
optimizer = ParallelAdam(committor.parameters(), lr=3e-3)
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
            bkecost = loss(grad_xs,invc,fwd_wl,bwrd_wl)
            cmcost = cmloss(i, dimer_sim.getConfig())
            cost = bkecost+cmcost
            cost.backward()
            
            optimizer.step()
            
            ftsoptimizer.step(configs[:,:6],len(configs),boxsize=box[0],remove_nullspace=dimer_nullspace)

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
