#Import necessarry tools from torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn as nn

#Import necessarry tools from tpstorch 
from tpstorch.ml.data import EXPReweightSimulation, EXPReweightSimulationManual, TSTValidation
from tpstorch.ml.optim import UnweightedSGD, EXPReweightSGD
from torch.distributed import distributed_c10d 
from mullerbrown import CommittorNet, MullerBrown, MullerBrownLoss
import numpy as np

dist.init_process_group(backend='mpi')
mpi_group = dist.distributed_c10d._get_default_group()

#Import any other thing
import tqdm, sys

prefix = 'simple'

#Initialize neural net
#Thinking about having Grant's initialization procedure...
committor = CommittorNet(d=2,num_nodes=200).to('cpu')

#Set initial configuration and BP simulator
start = torch.tensor([-1.2,0.9])
end = torch.tensor([-0.5,0.5])
# start = torch.tensor([[-0.5,1.5]])
# end = torch.tensor([[0.5,0.0]])
def initializer(s):
    return (1-s)*start+s*end
initial_config = initializer(dist.get_rank()/(dist.get_world_size()-1))
start = torch.tensor([[-0.5,1.5]])
end = torch.tensor([[0.5,0.0]])
mb_sim = MullerBrown(param="param",config=initial_config, rank=dist.get_rank(), dump=1, beta=0.1, kappa=10000, save_config=True, mpi_group = mpi_group, committor=committor)

#Generate unbiased configurations in reactant, product regions
n_boundary_samples = 100
react_data = torch.zeros(n_boundary_samples, start.shape[0]*start.shape[1], dtype=torch.float)
prod_data = torch.zeros(n_boundary_samples, end.shape[0]*end.shape[1], dtype=torch.float)
#Reactant
mb_sim_react = MullerBrown(param="param",config=start, rank=dist.get_rank(), dump=1, beta=0.1, kappa=0.0, save_config=True, mpi_group = mpi_group, committor=committor)
for i in range(n_boundary_samples):
    for j in range(500):
        mb_sim_react.step_unbiased()
    react_data[i] = mb_sim_react.getConfig()

#Product
mb_sim_prod = MullerBrown(param="param",config=end, rank=dist.get_rank(), dump=1, beta=0.1, kappa=0.0, save_config=True, mpi_group = mpi_group, committor=committor)
for i in range(n_boundary_samples):
    for j in range(500):
        mb_sim_prod.step_unbiased()
    prod_data[i] = mb_sim_prod.getConfig()

#Committor Loss
initloss = nn.MSELoss()
initoptimizer = UnweightedSGD(committor.parameters(), lr=1e-2)#,momentum=0.9,nesterov=True)#, weight_decay=1e-3)

#from torchsummary import summary
running_loss = 0.0
#Initial training try to fit the committor to the initial condition
for i in range(10**5):
    if i%1000 == 0:
        print("Init step "+str(i))
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
    #committor.project()
committor.zero_grad()

from torch.optim import lr_scheduler

#Construct EXPReweightSimulation
batch_size = 128
#dataset = EXPReweightSimulation(mb_sim, committor, period=10)
#loader = DataLoader(dataset,batch_size=batch_size)
datarunner = EXPReweightSimulationManual(mb_sim, committor, period=10, batch_size=batch_size, dimN=2)

#Optimizer, doing EXP Reweighting. We can do SGD (integral control), or Heavy-Ball (PID control)
loss = MullerBrownLoss(lagrange_bc = 400.0,batch_size=batch_size,start=start,end=end,radii=0.5,world_size=dist.get_world_size(),n_boundary_samples=n_boundary_samples,react_configs=react_data,prod_configs=prod_data)
if dist.get_rank() == 0:
    loss.compute_bc(committor, 0, 0)
#optimizer = EXPReweightSGD(committor.parameters(), lr=0.001, momentum=0.90, nesterov=True)
optimizer = UnweightedSGD(committor.parameters(), lr=0.001, momentum=0.9, nesterov=True)#, weight_decay=1e-3)

#lr_lambda = lambda epoch : 0.9**epoch
#scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)

loss_io = []
if dist.get_rank() == 0:
    loss_io = open("{}_loss.txt".format(prefix),'w')

#Training loop
#1 epoch: 200 iterations, 200 time-windows
for epoch in range(1):
    if dist.get_rank() == 0:
        print("epoch: [{}]".format(epoch+1))
    actual_counter = 0
    while actual_counter <= 20000:
        # if (actual_counter > 100) and (actual_counter < 10000):
            # loss.lagrange_bc += 0.01
            # if dist.get_rank() == 0:
                # print("lagrange_bc is now "+str(loss.lagrange_bc))
        
        # get data and reweighting factors
        config, grad_xs, invc, fwd_wl, bwrd_wl = datarunner.runSimulation()
        # print("CONFIGS")
        # print(grad_xs.size())
        # print(config.size())
        # print(invc.size())
        # print(fwd_wl.size())
        # print(bwrd_wl.size())
        # print("END CONFIGS")
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        cost = loss(grad_xs,committor,config,invc,fwd_wl,bwrd_wl,invc)
        cost.backward()
        # print(cost.size())
        #print("FACTORS")
        #print(fwd_wl)
        #print(bwrd_wl)
        #print(invc)
        #meaninvc, reweight = optimizer.step(fwd_weightfactors=fwd_wl, bwrd_weightfactors=bwrd_wl, reciprocal_normconstants=invc)
        optimizer.step()
        #print(meaninvc)
        #print(reweight)
        #committor.project()
        
        # print statistics
        with torch.no_grad():
            #if counter % 10 == 0:
            main_loss = loss.main_loss
            bc_loss = loss.bc_loss
            
            #What we need to do now is to compute with its respective weight
            #main_loss.mul_(reweight[dist.get_rank()])
            #bc_loss.mul_(reweight[dist.get_rank()])
            
            #All reduce the gradients
            #dist.all_reduce(main_loss)
            #dist.all_reduce(bc_loss)

            #Divide in-place by the mean inverse normalizing constant
            #main_loss.div_(meaninvc)
            #bc_loss.div_(meaninvc)
            # print("LOSS")
            # print(main_loss.size())
            # print(bc_loss.size())
            # print("END LOSS")
            
            #Print statistics 
            if dist.get_rank() == 0:
                print('[{}] loss: {:.5E} penalty: {:.5E} lr: {:.3E}'.format(actual_counter + 1, main_loss.item(), bc_loss.item(), optimizer.param_groups[0]['lr']))
                
                #Also print the reweighting factors
                print(loss.reweight)
                loss_io.write('{:d} {:.5E} \n'.format(actual_counter+1,main_loss))
                loss_io.flush()
                #Only save parameters from rank 0
                torch.save(committor.state_dict(), "{}_params_{}".format(prefix,dist.get_rank()+1))
        actual_counter += 1
    
##Perform Validation Test
if dist.get_rank() == 0:
    print("Finished Training! Now performing validation through committor analysis")
#Construct TSTValidation
print("Generating transition state")
for i in range(40000):
    config_cur = mb_sim.getConfig()
    mb_sim.step(committor_val=committor(config_cur), onlytst=True)

init_config = mb_sim.getConfig()
print("q value is "+str(committor(init_config)))
mb_sim = MullerBrown(param="param_tst",config=init_config, rank=dist.get_rank(), dump=1, beta=0.1, kappa=10000, save_config=True, mpi_group = mpi_group, committor=committor)
#mb_sim.setConfig(init_config)
#mb_sim = MullerBrown(param="param",config=init_config, rank=dist.get_rank(), dump=1, beta=0.20, kappa=80, save_config=True, mpi_group = mpi_group, committor=committor)
batch_size = 100 #batch of initial configuration to do the committor analysis per rank
dataset = TSTValidation(mb_sim, committor, period=20)
loader = DataLoader(dataset,batch_size=batch_size)


#Save validation scores and 
myval_io = open("{}_validation_{}.txt".format(prefix,dist.get_rank()+1),'w')
def myprod_checker(config):
    end = torch.tensor([[0.5,0.0]])
    end_2 = torch.tensor([[0.0,0.5]])
    radii = 0.3
    end_ = config-end
    end_ = end_.pow(2).sum()**0.5
    end_2_ = config-end_2
    end_2_ = end_2_.pow(2).sum()**0.5
    if ((end_ <= radii) or (end_2_ <= radii) or (config[1]<(config[0]+0.8))):
        return True
    else:
        return False
def myreact_checker(config):
    start = torch.tensor([[-0.5,1.5]])
    radii = 0.3
    start_ = config-start
    start_ = start_.pow(2).sum()**0.5
    if ((start_ <= radii) or (config[1]>(0.5*config[0]+1.5))):
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
    dataset.validate(batch, trials=100, validation_io=myval_io, product_checker=myprod_checker, reactant_checker=myreact_checker)
