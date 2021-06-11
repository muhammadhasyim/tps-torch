#Import necessarry tools from torch
import torch
import torch.nn as nn

#Import necessarry tools from tpstorch 
from tpstorch.ml.data import EXPReweightSimulation
from tpstorch.ml.optim import ParallelAdam, ParallelSGD
from tpstorch.ml.nn import BKELossEXP, BKELossFTS

#Import model-specific classes
from brownian_ml import CommittorNet, BrownianParticle
import numpy as np

#Save the rank and world size
from tpstorch import _rank, _world_size
rank = _rank
world_size = _world_size

#Import any other things
import tqdm, sys

torch.manual_seed(0)
np.random.seed(0)

prefix = 'vanilla'

#Set initial configuration and BP simulator
start = torch.tensor([[-1.0]])
end = torch.tensor([[1.0]])
def initializer(s):
    return (1-s)*start+s*end
initial_config = initializer(_rank/(_world_size-1))

kT = 1/15.0
bp_sampler = BrownianParticle(dt=5e-3,gamma=1.0,kT=kT, kappa=50,initial = initial_config,prefix=prefix,save_config=True)
bp_sampler_bc = BrownianParticle(dt=5e-3,gamma=1.0,kT=kT, kappa=0.0,initial = initial_config,prefix=prefix,save_config=True)

#Initialize neural net
committor = CommittorNet(d=1,num_nodes=200).to('cpu')
committor.load_state_dict(torch.load("initial_nn"))

#Construct EXPSimulation
batch_size = 4
datarunner = EXPReweightSimulation(bp_sampler, committor, period=100, batch_size=batch_size, dimN=1)

#Optimizer, doing EXP Reweighting. We can do SGD (integral control), or Heavy-Ball (PID control)
loss = BKELossEXP(  bc_sampler = bp_sampler_bc,
                    committor = committor,
                    lambda_A = 1e4,
                    lambda_B = 1e4,
                    start_react = start,
                    start_prod = end,
                    n_bc_samples = 100, 
                    bc_period = 10,
                    batch_size_bc = 0.5)

#optimizer = ParallelAdam(committor.parameters(), lr=1e-2)#, momentum=0.90,weight_decay=1e-3
optimizer = ParallelSGD(committor.parameters(), lr=5e-4,momentum=0.95)

#Save loss function statistics
loss_io = []
if _rank == 0:
    loss_io = open("{}_loss.txt".format(prefix),'w')

#Save timing statistics
import time
time_io = open("{}_timing_{}.txt".format(prefix,rank),'w')

#Training loop

for epoch in range(1):
    if _rank == 0:
        print("epoch: [{}]".format(epoch+1))
    actual_counter = 0
    while actual_counter <= 2500:
        
        t0 = time.time()
        # get data and reweighting factors
        config, grad_xs, invc, fwd_wl, bwrd_wl = datarunner.runSimulation()
        t1 = time.time()
        
        sampling_time = t1-t0

        t0 = time.time()
        
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        cost = loss(grad_xs,invc,fwd_wl,bwrd_wl)
        cost.backward()
        optimizer.step()
        
        t1 = time.time()
        optimization_time = t1-t0
        
        time_io.write('{:d} {:.5E} {:.5E} \n'.format(actual_counter+1,sampling_time, optimization_time))#main_loss.item(),bc_loss.item()))
        time_io.flush()
        
        # print statistics
        with torch.no_grad():
            main_loss = loss.main_loss
            bc_loss = loss.bc_loss
            
            if _rank == 0:
                #Print statistics 
                print('[{}] loss: {:.5E} penalty: {:.5E} lr: {:.3E}'.format(actual_counter + 1, main_loss.item(), bc_loss.item(), optimizer.param_groups[0]['lr']))
                #Also print the reweighting factors
                print(loss.zl)
                
                loss_io.write('{:d} {:.5E} {:.5E} \n'.format(actual_counter+1,main_loss.item(),bc_loss.item()))
                loss_io.flush()
                torch.save(committor.state_dict(), "{}_params_t_{}_{}".format(prefix,actual_counter,rank))
                #Only save parameters from rank 0
                torch.save(committor.state_dict(), "{}_params_{}".format(prefix,rank+1))
        actual_counter += 1
