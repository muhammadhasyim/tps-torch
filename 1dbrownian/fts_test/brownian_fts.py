#Import necessarry tools from torch
import torch
import torch.distributed as dist

#Import necessarry tools from tpstorch 
#import tpstorch.fts as fts
from tpstorch.fts import FTSSampler, AltFTSMethod
import numpy as np

#Import any other thing
import tqdm, sys

class BrownianParticle(FTSSampler):
    def __init__(self, dt, gamma, kT, initial,prefix='',save_config=False):
        super(BrownianParticle, self).__init__()
        
        #Timestep
        self.dt = dt

        #Noise variance
        self.coeff = np.sqrt(2*kT/gamma)
        self.gamma = gamma
        
        #The current position. We make sure that its gradient zero
        self.qt = initial.detach().clone()

        #IO for BP position and committor values
        self.save_config = save_config
        if self.save_config:
            self.qt_io = open("{}_bp_{}.txt".format(prefix,dist.get_rank()+1),"w")

        #Allocated value for self.qi
        self.invkT = 1/kT
        
        #Tracking steps
        self.timestep = 0

    #Runs dynamics after a given numver of n steps. Other parameters *_weight and *_bias defines the boundaries of the voronoi cell as hyperplanes.  
    def runSimulation(self, nsteps, left_weight, right_weight, left_bias, right_bias):
        for i in range(nsteps):
            q0 = self.qt-4*self.qt*(-1+self.qt**2)*self.dt/self.gamma + self.coeff*torch.normal(torch.tensor([[0.0]]),std=np.sqrt(self.dt))
            if dist.get_rank() == 0:
                if (torch.sum(q0*right_weight)+right_bias).item() < 0:
                    self.qt = q0.detach().clone()
            elif dist.get_rank() == dist.get_world_size()-1:
                if (torch.sum(q0*left_weight)+left_bias).item() > 0:
                    self.qt = q0.detach().clone()
            else:
                if (torch.sum(q0*left_weight)+left_bias).item() > 0 and (torch.sum(q0*right_weight)+right_bias).item() < 0:
                    self.qt = q0.detach().clone()
            self.timestep += 1

    #An unbiased simulation run
    def runUnbiased(self):
        q0 = self.qt-4*self.qt*(-1+self.qt**2)*self.dt/self.gamma + self.coeff*torch.normal(torch.tensor([[0.0]]),std=np.sqrt(self.dt))
        self.qt = q0.detach().clone()
    
    def getConfig(self):
        return self.qt.detach().clone()
    
    def dumpConfig(self):
        #Update timestep counter
        #self.timestep += 1
        if self.save_config:
        #    if self.timestep % 10 == 0:
            self.qt_io.write("{} {}\n".format(self.qt.item(),1/self.invkT))
            self.qt_io.flush()

#Override the class and modify the routine which dumps the transition path
class CustomFTSMethod(AltFTSMethod):
    def __init__(self,sampler,initial_config,final_config,num_nodes,deltatau,kappa):
        super(CustomFTSMethod, self).__init__(sampler,initial_config,final_config,num_nodes,deltatau,kappa)
    #Dump the string into a file
    def dump(self,dumpstring=False):
        if dumpstring:
            self.string_io.write("{} ".format(self.string[0,0]))
            self.string_io.write("\n")
        self.sampler.dumpConfig()
