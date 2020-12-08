#Import necessarry tools from torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn as nn

#Import necessarry tools from tpstorch 
from tpstorch.ml.optim import project_simplex
from tpstorch.ml import MLSamplerEXP
from tpstorch.ml.nn import CommittorLossEXP
import numpy as np

#Import any other thing
import tqdm, sys

dist.init_process_group(backend='mpi')

class CommittorNet(nn.Module):
    def __init__(self, d, num_nodes, unit=torch.sigmoid):
        super(CommittorNet, self).__init__()
        self.num_nodes = num_nodes
        self.d = d
        self.unit = unit
        self.lin1 = nn.Linear(d, num_nodes, bias=True)
        self.lin2 = nn.Linear(num_nodes, 1, bias=False)
        self.project()
        self.broadcast()

    def forward(self, x):
        x = self.lin1(x)
        x = self.unit(x)
        x = self.lin2(x)
        return x
    
    def broadcast(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                dist.broadcast(param.data,src=0)
    def project(self):
        #Project the coefficients so that they are make the output go from zero to one
        with torch.no_grad():
            self.lin2.weight.data = project_simplex(self.lin2.weight.data)

class BrownianParticle(MLSamplerEXP):
    def __init__(self, dt, gamma, kT, initial,prefix='',save_config=False):
        super(BrownianParticle, self).__init__(initial.detach().clone())
        
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
        self.kappa = 80
        
        #Tracking steps
        self.timestep = 0

        #Save config size and its flattened version
        self.config_size = initial.size()
        self.flattened_size = initial.flatten().size()
   
    def initialize_from_torchconfig(self, config):
        #by implementation, torch config cannot be structured: it must be a flat 1D tensor
        #config can ot be flat 
        if config.size() != self.flattened_size:
            raise RuntimeError("Config is not flat! Check implementation")
        else:
            self.qt = config.view(-1,1);
            if self.qt.size() != self.config_size:
                raise RuntimeError("New config has inconsistent size compared to previous simulation! Check implementation")
    
    def computeWForce(self, committor_val, qval):
        return -self.dt*self.kappa*self.torch_config.grad.data*(committor_val-qval)/self.gamma 
    
    def step(self, committor_val, onlytst=False):
        with torch.no_grad():
            #Update one step
            if onlytst:
                self.qt += self.computeWForce(committor_val,0.5)-4*self.qt*(-1+self.qt**2)*self.dt/self.gamma + self.coeff*torch.normal(torch.tensor([[0.0]]),std=np.sqrt(self.dt))
            else:
                self.qt += self.computeWForce(committor_val, self.qvals[dist.get_rank()])-4*self.qt*(-1+self.qt**2)*self.dt/self.gamma + self.coeff*torch.normal(torch.tensor([[0.0]]),std=np.sqrt(self.dt))
            
        #Don't forget to zero out gradient data after every timestep
        self.torch_config = (self.qt.flatten()).detach().clone()
        self.torch_config.requires_grad_()
        try:
            self.torch.grad.data.zero_()
        except:
            pass

    def save(self):
        #Update timestep counter
        self.timestep += 1
        if self.save_config:
            if self.timestep % 100 == 0:
                self.qt_io.write("{} {}\n".format(self.torch_config[0],1/self.invkT))
                self.qt_io.flush()
    
    def step_unbiased(self):
        #Update one 
        self.qt += -4*self.qt*(-1+self.qt**2)*self.dt/self.gamma + self.coeff*torch.normal(torch.tensor([[0.0]]),std=np.sqrt(self.dt))
        
        #Don't forget to zero out gradient data after every timestep
        self.torch_config = (self.qt.flatten()).detach().clone()
        self.torch_config.requires_grad_()
        try:
            self.torch.grad.data.zero_()
        except:
            pass

class BrownianLoss(CommittorLossEXP):
    def __init__(self, lagrange_bc, batch_size):
        super(BrownianLoss,self).__init__()
        self.lagrange_bc = lagrange_bc

    def compute_bc(self, committor, configs, invnormconstants):
        #Assume that first dimension is batch dimension
        loss_bc = torch.zeros(1)
        for i, config in enumerate(configs):
            if config.item() <= -1:
                loss_bc += 0.5*self.lagrange_bc*(committor(config.flatten())**2)*invnormconstants[i]
            if config.item() >= 1:
                loss_bc += 0.5*self.lagrange_bc*(committor(config.flatten())-1.0)**2*invnormconstants[i]
        return loss_bc/(i+1)
