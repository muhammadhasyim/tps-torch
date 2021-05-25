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

prefix = 'vanilla_highT'

#Set initial configuration and BP simulator
start = torch.tensor([[-1.0]])
end = torch.tensor([[1.0]])
def initializer(s):
    return (1-s)*start+s*end
initial_config = initializer(_rank/(_world_size-1))

kT = 1/15.0
bp_sampler = BrownianParticle(dt=2e-3,gamma=1.0,kT=kT, kappa=80,initial = initial_config,prefix=prefix,save_config=True)
bp_sampler_bc = BrownianParticle(dt=2e-3,gamma=1.0,kT=kT, kappa=0.0,initial = initial_config,prefix=prefix,save_config=True)

#Initialize neural net
committor = CommittorNet(d=1,num_nodes=200).to('cpu')

#Committor Loss for initialization
initloss = nn.MSELoss()
initoptimizer = ParallelSGD(committor.parameters(), lr=5e-2)

running_loss = 0.0

for i in tqdm.tqdm(range(10**3)):
    # zero the parameter gradients
    initoptimizer.zero_grad()
    
    # forward + backward + optimize
    q_vals = committor(initial_config.view(-1,1))
    targets = torch.ones_like(q_vals)*rank/(_world_size-1)
    cost = initloss(q_vals, targets)
    cost.backward()
    
    #Stepping up
    initoptimizer.step()

torch.save(committor.state_dict(), "{}_params_{}".format(prefix,_rank+1))

#Construct EXPSimulation
batch_size = 4
datarunner = EXPReweightSimulation(bp_sampler, committor, period=732, batch_size=batch_size, dimN=1)

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
optimizer = ParallelSGD(committor.parameters(), lr=5e-4,momentum=0.95,nesterov=True)
#optimizer = ParallelSGD(committor.parameters(), lr=5e-4,momentum=0.95,nesterov=True)

#Save loss function statistics
loss_io = []
if _rank == 0:
    loss_io = open("{}_loss.txt".format(prefix),'w')

#Training loop
for epoch in range(1):
    if _rank == 0:
        print("epoch: [{}]".format(epoch+1))
    actual_counter = 0
    while actual_counter <= 100:
        # get data and reweighting factors
        config, grad_xs, invc, fwd_wl, bwrd_wl = datarunner.runSimulation()
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        cost = loss(grad_xs,invc,fwd_wl,bwrd_wl)
        cost.backward()
        optimizer.step()
        
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
