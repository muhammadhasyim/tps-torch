#Import necessarry tools from torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn as nn

#Import necessarry tools from tpstorch 
from tpstorch.ml.data import EXPReweightSimulation
from tpstorch.ml.optim import UnweightedSGD, EXPReweightSGD
from brownian_ml import CommittorNet, BrownianParticle, BrownianLoss
import numpy as np

#Import any other thing
import tqdm, sys

prefix = 'lowT'
finalT = 0.15
initialT = 0.75

#Set initial configuration and BP simulator
start = torch.tensor([[-1.0]])
end = torch.tensor([[1.0]])
def initializer(s):
    return (1-s)*start+s*end
initial_config = initializer(dist.get_rank()/(dist.get_world_size()-1))
bp_sampler = BrownianParticle(dt=2e-3,gamma=1.0,kT=initialT,initial = initial_config,prefix=prefix)

#Initialize neural net
committor = CommittorNet(d=1,num_nodes=200).to('cpu')

#Committor Loss
initloss = nn.MSELoss()
initoptimizer = UnweightedSGD(committor.parameters(), lr=1e-2)#,momentum=0.9,nesterov=True)#, weight_decay=1e-3)

#from torchsummary import summary
running_loss = 0.0
#Initial training
#for counter, batch in tqdm.tqdm(enumerate(loader)):
for i in tqdm.tqdm(range(10**4)):
    # zero the parameter gradients
    initoptimizer.zero_grad()
    # forward + backward + optimize
    q_vals = committor(initial_config)
    targets = torch.ones_like(q_vals)*dist.get_rank()/(dist.get_world_size()-1)
    cost = initloss(q_vals, targets)#,committor,config,cx)
    cost.backward()
    #Stepping up
    initoptimizer.step()
    #committor.renormalize()
    committor.project()
committor.zero_grad()

from torch.optim import lr_scheduler

#Construct EXPReweightSimulation
batch_size = 16
dataset = EXPReweightSimulation(bp_sampler, committor, period=10)
loader = DataLoader(dataset,batch_size=batch_size)

#Optimizer, doing EXP Reweighting, use momentum because vanilla SGD is slow!
loss = BrownianLoss(lagrange_bc = 100.0,batch_size=batch_size)
optimizer = EXPReweightSGD(committor.parameters(), lr=0.1, momentum=0.9)

lr_lambda = lambda epoch : 0.9**epoch
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)

loss_io = []
if dist.get_rank() == 0:
    loss_io = open("{}_loss.txt".format(prefix),'w')

actual_counter = 0
for epoch in range(100):
    if dist.get_rank() == 0:
        print("epoch: [{}]".format(epoch+1))
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
            if counter % 10 == 0:
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
                    print('[{}] loss: {:.5E} penalty: {:.5E} lr: {:.3E}'.format(counter + 1, main_loss.item(), bc_loss.item(), optimizer.param_groups[0]['lr']))
                    print(reweight)
                    loss_io.write('{:d} {:.5E} \n'.format(actual_counter+1,main_loss))
                    loss_io.flush()
                torch.save(committor.state_dict(), "{}_params_{}".format(prefix,dist.get_rank()+1))
            actual_counter += 1
        
    #Decrease temperature at the same schedule as my learning rate
    if bp_sampler.invkT < 1/finalT:
        bp_sampler.invkT *= 1/0.9
    else:
        bp_sampler.invkT = 1/finalT
    scheduler.step()
