import torch.distributed as dist
dist.init_process_group(backend='mpi')
mygroup = dist.distributed_c10d._get_default_group()

import torch
import numpy as np
import tpstorch.fts as fts

#Create a sampler for FTS, performing MD simulation of a 1D brownian particle on a double well potential
class BrownianParticle(fts.FTSSampler):
    def __init__(self, dt, D, initial):
        super(BrownianParticle, self).__init__()
        self.dt = dt
        self.sigma = np.sqrt(2*D*dt)
        self.qt = initial.detach().clone()
        self.qt_io = open("bp_{}.txt".format(dist.get_rank()+1),"w")
    def runSimulation(self, nsteps, weights, biases):
        for i in range(nsteps):
            q0 = self.qt-4*self.qt*(-1+self.qt**2)*self.dt+torch.normal(torch.tensor([[0.0]]), self.sigma)
            if dist.get_rank() == 0:
                if (torch.sum(q0*weights[0])+biases[0]).item() < 0:
                    self.qt = q0.detach().clone()
            elif dist.get_rank() == dist.get_world_size()-1:
                if (torch.sum(q0*weights[0])+biases[0]).item() > 0:
                    self.qt = q0.detach().clone()
            else:
                if (torch.sum(q0*weights[0])+biases[0]).item() > 0 and (torch.sum(q0*weights[1])+biases[1]).item() < 0:
                    self.qt = q0.detach().clone()
    def getConfig(self):
        return self.qt.detach().clone()
    def dumpConfig(self):
        for item in self.qt:
            self.qt_io.write("{} ".format(item.item()))
        self.qt_io.write("\n")

#Override the class and modigy the dump routine
class CustomFTSMethod(fts.AltFTSMethod):
    def __init__(self,sampler,initial_config,final_config,num_nodes,deltatau,kappa):
        super(CustomFTSMethod, self).__init__(sampler,initial_config,final_config,num_nodes,deltatau,kappa)
    #Dump the string into a file
    def dump(self,dumpstring=False):
        if dumpstring:
            self.string_io.write("{} ".format(self.string[0,0]))
            self.string_io.write("\n")
        self.sampler.dumpConfig()

#Starting and ending configuration.
start = torch.tensor([[-1.0]])
end = torch.tensor([[1.0]])
def initializer(s):
    return (1-s)*start+s*end
alphas = torch.linspace(0.0,1,dist.get_world_size()+2)[1:-1]
bp_simulator = BrownianParticle(dt=0.001,D=0.1, initial=initializer(alphas[dist.get_rank()]))

fts_method = CustomFTSMethod(sampler=bp_simulator,initial_config=start,final_config=end,num_nodes=dist.get_world_size()+2,deltatau=0.01,kappa=0.01)

import tqdm
#for i in tqdm.tqdm(range(1)):
for i in tqdm.tqdm(range(1000)):
    fts_method.run(1)
    if (i % 200):
        fts_method.dump(True)
