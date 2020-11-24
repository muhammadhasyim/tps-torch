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
    def __init__(self, dt, gamma, kT, initial,prefix=''):
        super(BrownianParticle, self).__init__(initial.detach().clone())
        
        #Timestep
        self.dt = dt

        #Noise variance
        self.coeff = np.sqrt(2*kT/gamma)
        self.gamma = gamma
        
        #The current position. We make sure that its gradient zero
        self.qt = initial.detach().clone()

        #IO for BP position and committor values 
        self.qt_io = open("{}_bp_{}.txt".format(prefix,dist.get_rank()+1),"w")
        self.committor_io = open("{}_q_{}.txt".format(prefix,dist.get_rank()+1),"w")

        #Allocated value for self.qi
        self.invkT = 1/kT
        self.kappa = 80
        
        #Tracking steps
        self.timestep = 0
        
    def step(self, committor_val):
        with torch.no_grad():
            #Update one step
            self.qt += -self.dt*self.kappa*self.torch_config.grad.data*(committor_val-self.qvals[dist.get_rank()])/self.gamma -4*self.qt*(-1+self.qt**2)*self.dt/self.gamma + self.coeff*torch.normal(torch.tensor([[0.0]]),std=np.sqrt(self.dt))
            
        #Don't forget to zero out gradient data after every timestep
        self.torch_config = (self.qt.flatten()).detach().clone()
        self.torch_config.requires_grad_()
        try:
            self.torch.grad.data.zero_()
        except:
            pass
        
        #Update timestep counter
        self.timestep += 1
        if self.timestep % 100 == 0:
            self.qt_io.write("{} {}\n".format(self.torch_config[0],1/self.invkT))
            self.committor_io.write("{} {}\n".format(committor_val.item(),self.qvals[dist.get_rank()]))
            self.qt_io.flush()
            self.committor_io.flush()

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
