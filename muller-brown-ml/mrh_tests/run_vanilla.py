#Import necessarry tools from torch
import tpstorch
import torch
import torch.distributed as dist
import torch.nn as nn

#Import necessarry tools from tpstorch 
from mb_ml import MullerBrown, CommittorNet
from tpstorch.ml.data import EXPReweightSimulation
from tpstorch.ml.optim import ParallelSGD, ParallelAdam
from tpstorch.ml.nn import BKELossEXP
import numpy as np

#Grag the MPI group in tpstorch
mpi_group = tpstorch._mpi_group
world_size = tpstorch._world_size
rank = tpstorch._rank

#Import any other thing
import tqdm, sys
torch.manual_seed(3046)
np.random.seed(3046)

prefix = 'vanilla'

#Initialize neural net
def initializer(s):
    return (1-s)*start+s*end
start = torch.tensor([-0.5,1.5])
end = torch.tensor([0.6,0.08])

#Initialize neural net
committor = CommittorNet(d=2,num_nodes=200).to('cpu')

initial_config = initializer(rank/(dist.get_world_size()-1))
start = torch.tensor([[-0.5,1.5]])
end = torch.tensor([[0.6,0.08]])

kT = 10.0
mb_sim = MullerBrown(param="param",config=initial_config, rank=rank, dump=1, beta=1/kT, save_config=True, mpi_group = mpi_group, kappa=20000, committor=committor)
mb_sim_bc = MullerBrown(param="param_bc",config=initial_config, rank=rank, dump=1, beta=1/kT, save_config=True, mpi_group = mpi_group, kappa=0, committor=committor)

#Initial Training Loss
initloss = nn.MSELoss()
initoptimizer = ParallelSGD(committor.parameters(), lr=1e-3)#,momentum=0.95, nesterov=True)

#from torchsummary import summary
running_loss = 0.0
#Initial training try to fit the committor to the initial condition
for i in range(10**3):
    # zero the parameter gradients
    initoptimizer.zero_grad()
    # forward + backward + optimize
    q_vals = committor(initial_config.view(-1,2))
    targets = torch.ones_like(q_vals)*rank/(dist.get_world_size()-1)
    cost = initloss(q_vals, targets)#,committor,config,cx)
    cost.backward()
    #Stepping up
    initoptimizer.step()
    if i%1000 == 0 and rank == 0:
        print("Init step "+str(i),cost)
if rank == 0:
    #Only save parameters from rank 0
    torch.save(committor.state_dict(), "{}_params_{}".format(prefix,rank+1))
committor.zero_grad()

n_boundary_samples = 100
batch_size = 16#32#128
period = 25
datarunner = EXPReweightSimulation(mb_sim, committor, period=period, batch_size=batch_size, dimN=2)

#Initialize main loss function and optimizers
loss = BKELossEXP(  bc_sampler = mb_sim_bc, 
                    committor = committor,
                    lambda_A = 1e4,
                    lambda_B = 1e4,
                    start_react = start, 
                    start_prod = end, 
                    n_bc_samples = 100, 
                    bc_period = 10,
                    batch_size_bc = 0.5)

#This is the new committor loss! Feel free to comment this out if you don't need it
#cmloss = FTSCommittorLoss(  fts_sampler = mb_sim,
#                            committor = committor,
#                            fts_layer=ftslayer, 
#                            dimN = 2,
#                            lambda_fts=1e-1,
#                            fts_start=200,  
#                            fts_end=2000,
#                            fts_max_steps=batch_size*period*4, #To estimate the committor, we'll run foru times as fast 
#                            fts_rate=4, #In turn, we will only sample more committor value estimate  after 4 iterations 
#                            fts_min_count=2000, #Minimum count so that simulation doesn't (potentially) run too long
#                            batch_size_fts=0.5,
#                            tol = 1e-6,
#                            mode = 'shift'
#                            )

optimizer = ParallelAdam(committor.parameters(), lr=1e-3)
#optimizer = ParallelSGD(committor.parameters(), lr=1e-4,momentum=0.95,nesterov=True)

loss_io = []
if rank == 0:
    loss_io = open("{}_loss.txt".format(prefix),'w')

#Training loop
#We can train in terms of epochs, but we will keep it in one epoch
for epoch in range(1):
    if rank == 0:
        print("epoch: [{}]".format(epoch+1))
    actual_counter = 0
    while actual_counter <= 20000:
        # get data and reweighting factors
        configs, grad_xs, invc, fwd_wl, bwrd_wl  = datarunner.runSimulation()
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # (2) Update the neural network
        cost = loss(grad_xs, invc, fwd_wl, bwrd_wl)
        
        totalcost = cost#+cmcost
        totalcost.backward()
        optimizer.step()
        
        # print statistics
        with torch.no_grad():
            #if counter % 10 == 0:
            main_loss = loss.main_loss
            bc_loss = loss.bc_loss
            
            #Print statistics 
            if rank == 0:
                print('[{}] main_loss: {:.5E} bc_loss: {:.5E} lr: {:.3E}'.format(actual_counter + 1, main_loss.item(), bc_loss.item(),  optimizer.param_groups[0]['lr']),flush=True)
                print(loss.zl)
                
                loss_io.write('{:d} {:.5E} {:.5E} \n'.format(actual_counter+1,main_loss.item(),bc_loss.item()))
                loss_io.flush()
                
                if actual_counter % 100 == 0:
                    torch.save(committor.state_dict(), "{}_params_t_{}_{}".format(prefix,actual_counter,rank))
                
                torch.save(committor.state_dict(), "{}_params_{}".format(prefix,rank+1))
        actual_counter += 1