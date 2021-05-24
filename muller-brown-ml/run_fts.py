#import sys
#sys.path.insert(0,"/global/home/users/muhammad_hasyim/tps-torch/build")

#Import necessarry tools from torch
import tpstorch
import torch
import torch.distributed as dist
import torch.nn as nn

#Import necessarry tools from tpstorch 
from mb_fts import MullerBrown, CommittorFTSNet
from tpstorch.ml.data import FTSSimulation
from tpstorch.ml.optim import ParallelSGD, ParallelAdam, FTSUpdate
from tpstorch.ml.nn import BKELossFTS
import numpy as np

#Grag the MPI group in tpstorch
mpi_group = tpstorch._mpi_group
world_size = tpstorch._world_size
rank = tpstorch._rank

#Import any other thing
import tqdm, sys
torch.manual_seed(0)
np.random.seed(0)

prefix = 'simple'

#Initialize neural net
def initializer(s):
    return (1-s)*start+s*end
start = torch.tensor([-0.5,1.5])
end = torch.tensor([0.6,0.08])

#Initialize neural net
committor = CommittorFTSNet(d=2,start=start,end=end,num_nodes=200, fts_size=world_size).to('cpu')

initial_config = initializer(rank/(dist.get_world_size()-1))
start = torch.tensor([[-0.5,1.5]])
end = torch.tensor([[0.6,0.08]])

kT = 10.0
mb_sim = MullerBrown(param="param",config=initial_config, rank=rank, dump=1, beta=1/kT, save_config=True, mpi_group = mpi_group, committor=committor)

mb_sim_bc = MullerBrown(param="param_bc",config=initial_config, rank=rank, dump=1, beta=1/kT, save_config=True, mpi_group = mpi_group, committor=committor)


#mb_sim_committor = MullerBrown(param="param_tst",config=initial_config, rank=rank, dump=1, beta=1/kT, save_config=True, mpi_group = mpi_group, committor=committor)


#Initial Training Loss
initloss = nn.MSELoss()
initoptimizer = ParallelSGD(committor.parameters(), lr=1e-1,momentum=0.9)

#from torchsummary import summary
running_loss = 0.0
#Initial training try to fit the committor to the initial condition
for i in range(2*10**3):
    if i%1000 == 0 and rank == 0:
        print("Init step "+str(i))
    # zero the parameter gradients
    initoptimizer.zero_grad()
    # forward + backward + optimize
    q_vals = committor(initial_config.view(-1,2))
    targets = torch.ones_like(q_vals)*rank/(dist.get_world_size()-1)
    cost = initloss(q_vals, targets)#,committor,config,cx)
    cost.backward()
    #Stepping up
    initoptimizer.step()
committor.zero_grad()

#Construct FTSSimulation
n_boundary_samples = 100
batch_size = 32
datarunner = FTSSimulation(mb_sim, committor, period=40, batch_size=batch_size, dimN=2)

#Initialize main loss function and optimizers
loss = BKELossFTS(  bc_sampler = mb_sim_bc, 
                    committor = committor,
                    lambda_A = 1e4,
                    lambda_B = 1e4,
                    start_react = start, 
                    start_prod = end, 
                    n_bc_samples = 100, 
                    bc_period = 10,
                    batch_size_bc = 0.5)

optimizer = ParallelAdam(committor.parameters(), lr=1e-2)#, momentum=0.90,weight_decay=1e-3
#optimizer = ParallelSGD(committor.parameters(), lr=1e-2,momentum=0.90,nesterov=True)
ftsoptimizer = FTSUpdate(committor.lin1.parameters(), deltatau=1e-2,momentum=0.9,nesterov=True,kappa=0.1)

loss_io = []
if rank == 0:
    loss_io = open("{}_loss.txt".format(prefix),'w')

#Training loop
#We can train in terms of epochs, but we will keep it in one epoch
#For now
for epoch in range(1):
    if rank == 0:
        print("epoch: [{}]".format(epoch+1))
    actual_counter = 0
    while actual_counter <= 20000:
        #Evaluate the lower and upper committor values bounds
        
        #Before running simulation always re-initialize the list of committor values
        #That restarin the simulations and make sure that starting confifguration
        #is till withing each cell
        #TO DO: move this to  the datarunner!
        with torch.no_grad():
            mb_sim.committor_list[0] = 0.0#-0.1
            mb_sim.committor_list[-1] = 1.0#1.1
            for i in range(dist.get_world_size()):
                mb_sim.rejection_count[i] = 0
                if i > 0:
                    mb_sim.committor_list[i]= committor(0.5*(committor.lin1.string[i-1]+committor.lin1.string[i])).item()
            inftscell = mb_sim.checkFTSCell(committor(mb_sim.getConfig().flatten()), rank, dist.get_world_size())
            if inftscell:
                pass
            else:
                mb_sim.setConfig(committor.lin1.string[rank])
                
        # get data and reweighting factors
        configs, grad_xs  = datarunner.runSimulation()
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # (2) Update the neural network
        cost = loss(grad_xs, mb_sim.rejection_count)
        cost.backward()
        optimizer.step()
        
        # (1) Update the string
        ftsoptimizer.step(configs)
        #committor.lin1.update_string(config, react_data,prod_data,batch_size=batch_size)#, committor)
        
        # print statistics
        with torch.no_grad():
            #if counter % 10 == 0:
            main_loss = loss.main_loss
            bc_loss = loss.bc_loss
            
            #Print statistics 
            if rank == 0:
                print('[{}] main_loss: {:.5E} bc_loss: {:.5E} lr: {:.3E}'.format(actual_counter + 1, main_loss.item(), bc_loss.item(), optimizer.param_groups[0]['lr']),flush=True)
                print(committor.lin1.string,flush=True)#,committor.lin1.avgconfig)            
                print(loss.zl)
                
                loss_io.write('{:d} {:.5E} \n'.format(actual_counter+1,main_loss))
                loss_io.flush()
                
                #What we need to do now is to compute with its respective weight
                #if actual_counter% 5 == 0:
                torch.save(committor.state_dict(), "{}_params_t_{}_{}".format(prefix,actual_counter,rank))
                
                #Only save parameters from rank 0
                torch.save(committor.state_dict(), "{}_params_{}".format(prefix,rank+1))
        actual_counter += 1
