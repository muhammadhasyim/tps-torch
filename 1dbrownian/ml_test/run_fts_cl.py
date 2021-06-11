#Import necessarry tools from torch
import torch
import torch.nn as nn

#Import necessarry tools from tpstorch 
from tpstorch.ml.data import FTSSimulation
from tpstorch.ml.optim import ParallelAdam, FTSUpdate, ParallelSGD
from tpstorch.ml.nn import BKELossFTS, FTSCommittorLoss, CommittorLoss2, FTSLayer
from brownian_ml_fts import CommittorNet, BrownianParticle
import numpy as np

from tpstorch import _rank, _world_size
from tpstorch import dist

world_size = _world_size
rank = _rank

#Import any other thing
import tqdm, sys

torch.manual_seed(0)
np.random.seed(0)

prefix = 'fts_cl'

#Set initial configuration and BP simulator
start = torch.tensor([[-1.0]])
end = torch.tensor([[1.0]])
def initializer(s):
    return (1-s)*start+s*end
initial_config = initializer(_rank/(_world_size-1))

#Initialize neural net
committor = CommittorNet(d=1,num_nodes=200).to('cpu')
#Initialize the string for FTS method
ftslayer = FTSLayer(react_config=start,prod_config=end,num_nodes=world_size).to('cpu')

kT = 1/15.0
bp_sampler = BrownianParticle(dt=5e-3,ftslayer=ftslayer , gamma=1.0,kT=kT, initial = initial_config,prefix=prefix,save_config=True)
bp_sampler_bc = BrownianParticle(dt=5e-3,ftslayer=ftslayer, gamma=1.0,kT=kT, initial = initial_config,prefix=prefix,save_config=True)

committor.load_state_dict(torch.load("initial_nn"))

#Construct EXPSimulation
batch_size = 4
period = 100
datarunner = FTSSimulation(bp_sampler, committor, period=period, batch_size=batch_size, dimN=1)
#,mode='adaptive',min_rejection_count=1)#,max_period=100)#,max_steps=10**3)#,max_period=10)

#Optimizer, doing EXP Reweighting. We can do SGD (integral control), or Heavy-Ball (PID control)
loss = BKELossFTS(  bc_sampler = bp_sampler_bc,
                    committor = committor,
                    lambda_A = 1e4,
                    lambda_B = 1e4,
                    start_react = start,
                    start_prod = end,
                    n_bc_samples = 100, 
                    bc_period = 10,
                    batch_size_bc = 0.5,
                    tol = 2e-9,
                    mode='shift')

cmloss = CommittorLoss2( cl_sampler = bp_sampler_bc,
                        committor = committor,
                        lambda_cl=100.0,
                        cl_start=10,
                        cl_end=5000,
                        cl_rate=40,
                        cl_trials=100,
                        batch_size_cl=0.5
                        )

#optimizer = ParallelAdam(committor.parameters(), lr=5e-3)#, momentum=0.90,weight_decay=1e-3
optimizer = ParallelSGD(committor.parameters(), lr=5e-4,momentum=0.95)#,nesterov=True)
ftsoptimizer = FTSUpdate(committor.lin1.parameters(), deltatau=1e-2,momentum=0.9,nesterov=True,kappa=0.1)

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
        config, grad_xs = datarunner.runSimulation()
        t1 = time.time()
        
        sampling_time = t1-t0
        
        t0 = time.time()
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # (2) Update the neural network
        cost = loss(grad_xs, bp_sampler.rejection_count)
        t1 = time.time()
        optimization_time = t1-t0

        t0 = time.time()
        cmcost = cmloss(actual_counter, bp_sampler.getConfig())
        t1 = time.time()
        supervised_time = t1-t0
        
        t0 = time.time()
        totalcost = cost+cmcost
        totalcost.backward()
        optimizer.step()
        t1 = time.time()
        optimization_time += t1-t0
        
        time_io.write('{:d} {:.5E} {:.5E} {:.5E} \n'.format(actual_counter+1,sampling_time, optimization_time, supervised_time))#main_loss.item(),bc_loss.item()))
        time_io.flush()
        
        # print statistics
        with torch.no_grad():
            main_loss = loss.main_loss
            bc_loss = loss.bc_loss
            
            #Track the average number of sampling period
            test = torch.tensor([float(datarunner.period)])
            dist.all_reduce(test)
            test /= float(world_size)
            if rank == 0:
                print(test)
            
            if _rank == 0:
                #Print statistics 
                print('[{}] main_loss: {:.5E} bc_loss: {:.5E} fts_loss: {:.5E} lr: {:.3E} period: {:.3f}'.format(actual_counter + 1, main_loss.item(), bc_loss.item(), cmcost.item(), optimizer.param_groups[0]['lr'], test.item()),flush=True)
                print(ftslayer.string,flush=True)
                
                #Also print the reweighting factors
                print(loss.zl)
                
                loss_io.write('{:d} {:.5E} {:.5E} {:.5E} \n'.format(actual_counter+1,main_loss.item(),bc_loss.item(),cmcost.item()))
                loss_io.flush()
                
                torch.save(committor.state_dict(), "{}_params_t_{}_{}".format(prefix,actual_counter,rank))
                torch.save(ftslayer.state_dict(), "{}_string_t_{}_{}".format(prefix,actual_counter,rank))
                
                torch.save(committor.state_dict(), "{}_params_{}".format(prefix,rank+1))
                torch.save(ftslayer.state_dict(), "{}_string_{}".format(prefix,rank+1))
        #scheduler.step()
        actual_counter += 1
