import sys
sys.path.append("..")
import time
t0 = time.time()

#Import necessarry tools from torch
import tpstorch
import torch
import torch.distributed as dist
import torch.nn as nn
import scipy.spatial

#Import necessarry tools from tpstorch 
from dimer_ftsus_nosolv import DimerFTSUS
from committor_nn import initializeConfig, CommittorNet, CommittorNetBP, CommittorNetDR, SchNet
from dimer_ftsus_nosolv import FTSLayerUSCustom as FTSLayer
from tpstorch.ml.data import FTSSimulation, EXPReweightStringSimulation
from tpstorch.ml.optim import ParallelSGD, ParallelAdam
from tpstorch.ml.nn import BKELossEXP, CommittorLoss2
import numpy as np

#Grag the MPI group in tpstorch
mpi_group = tpstorch._mpi_group
world_size = tpstorch._world_size
rank = tpstorch._rank

#Import any other thing
import tqdm, sys
# reload count
count = int(np.genfromtxt("count.txt"))
torch.manual_seed(count)
np.random.seed(count)

prefix = 'simple'

#(1) Initialization
r0 = 2**(1/6.0)
width =  0.25
Np = 30+2
box = [8.617738760127533, 8.617738760127533, 8.617738760127533]
kappa_perp = 1200.0
kappa_par = 1200.0
kT = 1.0

initial_config = np.genfromtxt("../restart/config_"+str(rank)+".xyz", usecols=(1,2,3))
start = np.genfromtxt("../restart_bc/config_"+str(rank)+"_react.xyz", usecols=(1,2,3))
end = np.genfromtxt("../restart_bc/config_"+str(rank)+"_prod.xyz", usecols=(1,2,3))
initial_config = torch.from_numpy(initial_config)
start = torch.from_numpy(start)
end = torch.from_numpy(end)
initial_config = initial_config.float()
start = start.float()
end = end.float()

#Initialize neural net
#committor = torch.jit.script(CommittorNetDR(num_nodes=2500, boxsize=box[0]).to('cpu'))
#committor = torch.jit.script(CommittorNetBP(num_nodes=200, boxsize=box[0], Np=32,rc=2.5,sigma=1.0).to('cpu'))
committor = SchNet(hidden_channels = 64, num_filters = 64, num_interactions = 3, num_gaussians = 50, cutoff = box[0], max_num_neighbors = 31, boxsize=box[0], Np=32, dim=3).to('cpu')

#Initialize the string for FTS method
ftslayer = FTSLayer(react_config=start[:2].flatten(),prod_config=end[:2].flatten(),num_nodes=world_size,boxsize=box[0],kappa_perpend=kappa_perp, kappa_parallel=kappa_par, num_particles=2).to('cpu')
#Load the pre-initialized neural network and string
committor.load_state_dict(torch.load("simple_params", map_location=torch.device('cpu')))
ftslayer.load_state_dict(torch.load("../test_string_config"))
ftslayer.set_tangent()

n_boundary_samples = 100
batch_size = 8
period = 25
dimer_sim_bc = DimerFTSUS(  param="param_bc",
                            config=initial_config.clone().detach(), 
                            rank=rank, 
                            beta=1/kT, 
                            kappa = 0.0, 
                            save_config=False, 
                            mpi_group = mpi_group, 
                            ftslayer=ftslayer,
                            output_time=batch_size*period
                            )
dimer_sim = DimerFTSUS( param="param",
                        config=initial_config.clone().detach(), 
                        rank=rank, 
                        beta=1/kT, 
                        kappa = kappa_perp, 
                        save_config=True, 
                        mpi_group = mpi_group, 
                        ftslayer=ftslayer,
                        output_time=batch_size*period
                        )
dimer_sim.useRestart()
dimer_sim_com = DimerFTSUS(  param="param",
                            config=initial_config.clone().detach(), 
                            rank=rank, 
                            beta=1/kT, 
                            kappa = 0.0, 
                            save_config=False, 
                            mpi_group = mpi_group, 
                            ftslayer=ftslayer,
                            output_time=batch_size*period
                            )

#Construct datarunner
datarunner = EXPReweightStringSimulation(dimer_sim, committor, period=period, batch_size=batch_size, dimN=Np*3)
#Construct optimizers
optimizer = ParallelAdam(committor.parameters(), lr=1e-4)
optimizer.load_state_dict(torch.load("optimizer_params"))

#Initialize main loss function and optimizers
#Optimizer, doing EXP Reweighting. We can do SGD (integral control), or Heavy-Ball (PID control)
loss = BKELossEXP(  bc_sampler = dimer_sim_bc,
                    committor = committor,
                    lambda_A = 1e4,
                    lambda_B = 1e4,
                    start_react = start,
                    start_prod = end,
                    n_bc_samples = 0, 
                    bc_period = 100,
                    batch_size_bc = 0.5,
                    )

cmloss = CommittorLoss2( cl_sampler = dimer_sim_com,
                        committor = committor,
                        lambda_cl=100.0,
                        cl_start=10,
                        cl_end=5000,
                        cl_rate=10,
                        cl_trials=100,
                        batch_size_cl=0.5
                        )


loss_io = []
if rank == 0:
    loss_io = open("{}_statistic_{}.txt".format(prefix,rank+1),'a')

lambda_cl_end = 10**3
cl_start=200
cl_end=10000
cl_stepsize = (lambda_cl_end-cmloss.lambda_cl)/(cl_end-cl_start)

# Save reactant, product configurations
loss.react_configs = torch.load("react_configs_"+str(rank+1)+".pt")
loss.prod_configs = torch.load("prod_configs_"+str(rank+1)+".pt")
loss.n_bc_samples = torch.load("n_bc_samples_"+str(rank+1)+".pt")
# cmloss variables
cmloss.lambda_cl = torch.load("lambda_cl_"+str(rank+1)+".pt")
cmloss.cl_configs = torch.load("cl_configs_"+str(rank+1)+".pt")
cmloss.cl_configs_values = torch.load("cl_configs_values_"+str(rank+1)+".pt")
cmloss.cl_configs_count = torch.load("cl_configs_count_"+str(rank+1)+".pt")

#Training loop
for epoch in range(1):
    if rank == 0:
        print("epoch: [{}]".format(epoch+1))
    time_max = 9.0*60
    time_out = True
    #for i in range(count,count+100):#20000)):
    i = count
    while(time_out):
        if (i > cl_start) and (i <= cl_end):
            cmloss.lambda_cl += cl_stepsize
        elif i > cl_end:
            cmloss.lambda_cl = lambda_cl_end
        # get data and reweighting factors
        config, grad_xs, invc, fwd_wl, bwrd_wl = datarunner.runSimulation()
        dimer_sim.dumpRestart()
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # (2) Update the neural network
        # forward + backward + optimize
        bkecost = loss(grad_xs,invc,fwd_wl,bwrd_wl)
        cmcost = cmloss(i, dimer_sim.getConfig())
        cost = bkecost+cmcost
        cost.backward()
        
        optimizer.step()

        # print statistics
        with torch.no_grad():
            main_loss = loss.main_loss
            cm_loss = cmloss.cl_loss
            bc_loss = loss.bc_loss
            
            #Print statistics 
            if rank == 0:
                if i%100 == 0:
                    torch.save(committor.state_dict(), "{}_params_t_{}_{}".format(prefix,i,rank))
                torch.save(committor.state_dict(), "{}_params".format(prefix))
                torch.save(optimizer.state_dict(), "optimizer_params")
                np.savetxt("count.txt", np.array((i+1,)))
                loss_io.write('{:d} {:.5E} {:.5E} {:.5E}\n'.format(i+1,main_loss.item(),bc_loss.item(),cm_loss.item()))
                loss_io.flush()
            torch.save(cmloss.lambda_cl, "lambda_cl_"+str(rank+1)+".pt")
            torch.save(cmloss.cl_configs, "cl_configs_"+str(rank+1)+".pt")
            torch.save(cmloss.cl_configs_values, "cl_configs_values_"+str(rank+1)+".pt")
            torch.save(cmloss.cl_configs_count, "cl_configs_count_"+str(rank+1)+".pt")
            i = i+1
            t1 = time.time()
            time_diff = t1-t0
            time_diff = torch.tensor(time_diff)
            dist.all_reduce(time_diff,op=dist.ReduceOp.MAX)
            if time_diff > time_max:
                time_out = False
