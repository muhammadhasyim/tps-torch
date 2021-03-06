#Start up torch dist package
import torch
import torch.distributed as dist
dist.init_process_group(backend='mpi')

#Load classes for simulations and controls
from brownian_fts import BrownianParticle
import numpy as np

#Starting and ending configuration.
start = torch.tensor([[-1.0]])
end = torch.tensor([[1.0]])
def initializer(s):
    return (1-s)*start+s*end
alphas = torch.linspace(0.0,1,dist.get_world_size()+2)[1:-1]

bp_simulator = BrownianParticle(dt=2e-3,gamma=1.0, kT = 0.4, initial=initializer(alphas[dist.get_rank()]),prefix='test',save_config=False)

#Generate data for validation test
#For this 1D brownian example case, the TSE is generated by the middle replica.
#Note that This is assuming that you run an odd number of MPI processes
data = np.loadtxt('test_bp_{}.txt'.format(int((dist.get_world_size()+1)/2)))

#If I run 10 processes, that's 10*10 = 100 initial configurations!
num_configs = 10
trials = 500
validation_io = open("test_validation_{}.txt".format(dist.get_rank()+1),"w")
import tqdm
#For loop over initial states
if dist.get_rank() == 0:
    print("Ready to generate validation test!".format(dist.get_rank()+1))
for i in range(num_configs):
    counts = []
    initial_config = torch.from_numpy(np.array([[np.random.choice(data[:,0])]]).astype(np.float32)).detach().clone()    
    for j in tqdm.tqdm(range(trials)):
        hitting = False
        bp_simulator.qt = initial_config.detach().clone()
        #Run simulation and stop until it falls into the product or reactant state
        while hitting is False:
            bp_simulator.runUnbiased()
            if np.abs(bp_simulator.qt.item()) >= 1.0:
                if bp_simulator.qt.item() < 0:
                    counts.append(0.0)
                elif bp_simulator.qt.item() > 0:
                    counts.append(1.0)
                hitting = True
    #Compute the committor after a certain number of trials
    counts = np.array(counts)
    mean_count = np.mean(counts) 
    conf_count = 1.96*np.std(counts)/len(counts)**0.5 #do 95 % confidence interval
    
    #Save into io
    if validation_io is not None:
        validation_io.write('{} {} {} \n'.format(mean_count, conf_count,initial_config.item()))
        validation_io.flush()

