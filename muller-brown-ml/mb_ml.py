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

class MullerBrown(MLSamplerEXP):
    #I'm wrapping the sampler in a sampler
    #Lord praise have mercy on my soul
    def __init__(self, sampler):
        super(MullerBrown, self).__init(sampler)
    
    def initialize_from_torchconfig(self, config):

    def step(self, committor_val, onlytst=False):

    def step_unbiased(self):

    def step_unbiased(self):

class MullerBrownLoss(CommittorLossEXP):
    def __init__(self, lagrange_bc, batch_size):
        super(MullerBrownLoss,self).__init__()
        self.lagrange_bc = lagrange_bc

    def compute_bc(self, committor, configs, invnormconstants):
        #Assume that first dimension is batch dimension
        loss_bc = torch.zeros(1)
        for i, config in enumerate(configs):
            print("CONFIGS PRINTING")
            print(config)
            if config.item() <= -1:
                loss_bc += 0.5*self.lagrange_bc*(committor(config.flatten())**2)*invnormconstants[i]
            if config.item() >= 1:
                loss_bc += 0.5*self.lagrange_bc*(committor(config.flatten())-1.0)**2*invnormconstants[i]
        return loss_bc/(i+1)
