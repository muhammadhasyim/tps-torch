#Import necessarry tools from torch
import torch
import torch.nn as nn

#Import necessarry tools from tpstorch 
from tpstorch.ml.data import FTSSimulation
from tpstorch.ml.optim import ParallelAdam, FTSUpdate, ParallelSGD
from tpstorch.ml.nn import BKELossFTS
from brownian_ml_fts import CommittorFTSNet, BrownianParticle
import numpy as np

from tpstorch import dist, _rank, _world_size

world_size = _world_size
rank = _rank

#Import any other thing
import tqdm, sys
torch.manual_seed(0)
np.random.seed(0)

prefix = 'fts_highT'

#Set initial configuration and BP simulator
start = torch.tensor([[-1.0]])
end = torch.tensor([[1.0]])
def initializer(s):
    return (1-s)*start+s*end
initial_config = initializer(_rank/(_world_size-1))

#Initialize neural net
committor = CommittorFTSNet(d=1,start=start.flatten(),end=end.flatten(),num_nodes=200, fts_size=world_size).to('cpu')

kT = 1/15.0#1/15.0#1/20

bp_sampler = BrownianParticle(dt=1e-2,committor = committor , gamma=1.0,kT=kT, initial = initial_config,prefix=prefix,save_config=True)
bp_sampler_bc = BrownianParticle(dt=1e-2,committor = committor, gamma=1.0,kT=kT, initial = initial_config,prefix=prefix,save_config=True)


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
    cost = initloss(q_vals, targets)#,committor,config,cx)
    cost.backward()
    
    #Stepping up
    initoptimizer.step()
    torch.save(committor.state_dict(), "{}_params_{}".format(prefix,_rank+1))

#Construct EXPSimulation
batch_size = 4
datarunner = FTSSimulation(bp_sampler, committor, period=1, batch_size=batch_size, dimN=1,min_rejection_count=5)

#Optimizer, doing EXP Reweighting. We can do SGD (integral control), or Heavy-Ball (PID control)
loss = BKELossFTS(  bc_sampler = bp_sampler_bc,
                    committor = committor,
                    lambda_A = 1e4,
                    lambda_B = 1e4,
                    start_react = start,
                    start_prod = end,
                    n_bc_samples = 100, 
                    bc_period = 10,
                    batch_size_bc = 0.5)

optimizer = ParallelAdam(committor.parameters(), lr=1e-2)
#optimizer = ParallelSGD(committor.parameters(), lr=5e-4,momentum=0.95,nesterov=True)
ftsoptimizer = FTSUpdate(committor.lin1.parameters(), deltatau=1e-2,momentum=0.9,nesterov=True,kappa=0.1)

#Save loss function statistics
loss_io = []
if _rank == 0:
    loss_io = open("{}_loss.txt".format(prefix),'w')

#Training loop
for epoch in range(1):
    if _rank == 0:
        print("epoch: [{}]".format(epoch+1))
    actual_counter = 0
    while actual_counter <= 2000:
        #Before running simulation always re-initialize the list of committor values
        #That restarin the simulations and make sure that starting confifguration
        #is till withing each cell
        #TO DO: move this to  the datarunner!
        with torch.no_grad():
            bp_sampler.committor_list[0] = 0.0#-0.1
            bp_sampler.committor_list[-1] = 1.0#1.1
            bp_sampler.steps = 0.0
            for i in range(_world_size):
                bp_sampler.rejection_count[i] = 0
                if i > 0:
                    bp_sampler.committor_list[i]= committor(0.5*(committor.lin1.string[i-1]+committor.lin1.string[i])).item()
            inftscell = bp_sampler.checkFTSCell(committor(bp_sampler.getConfig().flatten()), rank, _world_size)
            if inftscell:
                pass
            else:
                bp_sampler.setConfig(committor.lin1.string[rank])
        
        # get data and reweighting factors
        config, grad_xs = datarunner.runSimulation()
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        cost = loss(grad_xs, bp_sampler.rejection_count)
        cost.backward()
        optimizer.step()
        
        # print statistics
        with torch.no_grad():
            main_loss = loss.main_loss
            bc_loss = loss.bc_loss
            
            #Track the average number of sampling period
            test = torch.tensor([float(datarunner.period)])
            dist.all_reduce(test)
            test /= float(_world_size)

            if _rank == 0:
                #Print statistics 
                print('[{}] loss: {:.5E} penalty: {:.5E} lr: {:.3E} period: {:.3E}'.format(actual_counter + 1, main_loss.item(), bc_loss.item(), optimizer.param_groups[0]['lr'],test.item()))
                
                #Also print the reweighting factors
                print(loss.zl)
                
                loss_io.write('{:d} {:.5E} {:.5E} \n'.format(actual_counter+1,main_loss.item(),bc_loss.item()))
                loss_io.flush()
                torch.save(committor.state_dict(), "{}_params_t_{}_{}".format(prefix,actual_counter,rank))
                #Only save parameters from rank 0
                torch.save(committor.state_dict(), "{}_params_{}".format(prefix,rank+1))
        actual_counter += 1
