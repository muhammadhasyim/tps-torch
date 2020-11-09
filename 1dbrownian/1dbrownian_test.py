#Quick 1D Test for 1D Brownian motion
#Muhammad R. Hasyim

import numpy as np
import torch

#Start off MPI with Torch's distributed package
import torch.distributed as dist
dist.init_process_group(backend='mpi')

from tpstorch import fts

#Does brownian motion on a double well potential
class BrownianParticle:
    def __init__(self, dt, D, initial):
        self.dt = dt
        self.sigma = np.sqrt(2*D*dt)
        self.qt = initial.detach().clone()
        self.qt_io = open("bp_{}.txt".format(dist.get_rank()+1),"w")
    
    def run(self, nsteps, weights, biases):
        for i in range(nsteps):
            q0 = self.qt-4*self.qt*(-1+self.qt**2)*self.dt+torch.normal(torch.tensor([0.0]), self.sigma)
            if torch.dot(q0,weights[0])+biases[0] >= 0 and torch.dot(q0,weights[1])+biases[1] <= 0:
                self.qt = q0.detach().clone()
    def get(self):
        return self.qt
    def dump(self):
        for item in self.qt:
            self.qt_io.write("{} ".format(item))
        self.qt_io.write("\n")

#Start and end on the two minima
start = torch.tensor([-1.0])
end = torch.tensor([1.0])

#Allocated nodal variables
alphas = torch.linspace(0.0,1,dist.get_world_size()+2)[1:-1]

#Helper function initializing brownian particle
def initializer(s):
    return (1-s)*start+s*end
bp_simulator = BrownianParticle(dt=0.001,D=1.0, initial=initializer(alphas[dist.get_rank()]))

#initialize the FTS Method class
fts_method = fts.FTSMethod(sampler=bp_simulator,initial_config=start,final_config=end,num_nodes=dist.get_world_size()+2,deltatau=0.01,kappa=0.0)

n_steps = 1 #defines how many timesteps for the Brownian motion
import tqdm
for i in tqdm.tqdm(range(2000)):
    fts_method.run(n_steps)
    if i % 100:
        fts_method.dump()
