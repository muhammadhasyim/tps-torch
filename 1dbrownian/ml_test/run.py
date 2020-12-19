#Append tpstorch path
import sys
sys.path.append('/global/common/software/m2893/tps-torch/build')

#Import necessarry tools from torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn as nn

#Import necessarry tools from tpstorch 
from tpstorch.ml.data import EXPReweightSimulation, TSTValidation
from tpstorch.ml.optim import UnweightedSGD, EXPReweightSGD
from brownian_ml import CommittorNet, BrownianParticle, BrownianLoss
import numpy as np

#Import any other thing
import tqdm, sys

prefix = 'highT'

#Set initial configuration and BP simulator
start = torch.tensor([[-1.0]])
end = torch.tensor([[1.0]])
def initializer(s):
    return (1-s)*start+s*end
initial_config = initializer(dist.get_rank()/(dist.get_world_size()-1))
bp_sampler = BrownianParticle(dt=2e-3,gamma=1.0,kT=0.4,initial = initial_config,prefix=prefix,save_config=True)

#Initialize neural net
committor = CommittorNet(d=1,num_nodes=100).to('cpu')

#Committor Loss for initialization
initloss = nn.MSELoss()
initoptimizer = UnweightedSGD(committor.parameters(), lr=1e-2)

running_loss = 0.0
#Initial training try to fit the committor to the initial condition
for i in tqdm.tqdm(range(10**3)):
    # zero the parameter gradients
    initoptimizer.zero_grad()
    # forward + backward + optimize
    q_vals = committor(initial_config)
    targets = torch.ones_like(q_vals)*dist.get_rank()/(dist.get_world_size()-1)
    cost = initloss(q_vals, targets)#,committor,config,cx)
    cost.backward()
    #Stepping up
    initoptimizer.step()
    #Add projection step
    committor.project()

committor.zero_grad()

from torch.optim import lr_scheduler

#Construct EXPReweightSimulation
batch_size = 128
dataset = EXPReweightSimulation(bp_sampler, committor, period=10)
loader = DataLoader(dataset,batch_size=batch_size)

#Optimizer, doing EXP Reweighting. We can do SGD (integral control), or Heavy-Ball (PID control)
loss = BrownianLoss(lagrange_bc = 100.0,batch_size=batch_size)
optimizer = EXPReweightSGD(committor.parameters(), lr=0.05, momentum=0.90)

#Save loss function statistics
loss_io = []
if dist.get_rank() == 0:
    loss_io = open("{}_loss.txt".format(prefix),'w')

#Training loop
#1 epoch: 200 iterations, 200 time-windows
for epoch in range(1):
    if dist.get_rank() == 0:
        print("epoch: [{}]".format(epoch+1))
    actual_counter = 0
    for counter, batch in enumerate(loader):
        if counter > 200:
            break
        
        # get data and reweighting factors
        config, grad_xs, invc, fwd_wl, bwrd_wl = batch
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        cost = loss(grad_xs,committor,config,invc)
        cost.backward()
        meaninvc, reweight = optimizer.step(fwd_weightfactors=fwd_wl, bwrd_weightfactors=bwrd_wl, reciprocal_normconstants=invc)
        committor.project()
        
        # print statistics
        with torch.no_grad():
            main_loss = loss.main_loss
            bc_loss = loss.bc_loss
            
            #What we need to do now is to compute with its respective weight
            main_loss.mul_(reweight[dist.get_rank()])
            bc_loss.mul_(reweight[dist.get_rank()])
            
            #All reduce the gradients
            dist.all_reduce(main_loss)
            dist.all_reduce(bc_loss)

            #Divide in-place by the mean inverse normalizing constant
            main_loss.div_(meaninvc)
            bc_loss.div_(meaninvc)
            
            if dist.get_rank() == 0:
                #Print statistics 
                print('[{}] loss: {:.5E} penalty: {:.5E} lr: {:.3E}'.format(counter + 1, main_loss.item(), bc_loss.item(), optimizer.param_groups[0]['lr']))
                
                #Also print the reweighting factors
                print(reweight)
                loss_io.write('{:d} {:.5E} \n'.format(actual_counter+1,main_loss))
                loss_io.flush()
                #Only save parameters from rank 0. If you want to save what it looks like per iteration, change the name so that it doesn't overwrite
                torch.save(committor.state_dict(), "{}_params_{}".format(prefix,dist.get_rank()+1))
        actual_counter += 1
    

##Perform Validation Test
if dist.get_rank() == 0:
    print("Finished Training! Now performing validation through committor analysis")

#Construct class that facilitates validation 
batch_size = 10 #batch of initial configuration to do the committor analysis per rank
dataset = TSTValidation(bp_sampler, committor, period=20)
loader = DataLoader(dataset,batch_size=batch_size)

data = np.loadtxt('{}_bp_6.txt'.format(prefix))[100::2,0]
val = float(data[dist.get_rank()+1])
init_config = torch.tensor([[val]]).flatten()
bp_sampler.initialize_from_torchconfig(init_config)

#File for validation scores
myval_io = open("{}_validation_{}.txt".format(prefix,dist.get_rank()+1),'w')

#Functions to check if they fall in product or reactant basins
def myprod_checker(config):
    if config >= 1.0:
        return True
    else:
        return False
def myreact_checker(config):
    if config <= -1.0:
        return True
    else:
        return False

#Run validation loop
actual_counter = 0
for epoch, batch in enumerate(loader):
    if epoch > 1:
        break
    if dist.get_rank() == 0:
        print("epoch: [{}]".format(epoch+1))
    
    #Call the validation function
    configs, committor_values = batch
    dataset.validate(batch, trials=500, validation_io=myval_io, product_checker=myprod_checker, reactant_checker=myreact_checker)
