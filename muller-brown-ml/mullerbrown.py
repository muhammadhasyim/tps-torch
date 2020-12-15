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
        #At the moment, x is flat. So if you want it to be 2x1 or 3x4 arrays, then you do it here!
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

# This is a sketch of what it should be like, but basically have to also make committor play nicely
# within function
# have omnious committor part that actual committor overrides?
class MullerBrown(MySampler):
    def __init__(self,param,config,rank,dump,beta,kappa):
        super(MullerBrown, self).__init__(param,config.detach().clone(),rank,dump,beta,kappa)

        self.committor = []

        #Save config size and its flattened version
        self.config_size = config.size()
        self.flattened_size = config.flatten().size()

    def initialize_from_torchconfig(self, config):
        #by implementation, torch config cannot be structured: it must be a flat 1D tensor
        #config can ot be flat 
        if config.size() != self.flattened_size:
            raise RuntimeError("Config is not flat! Check implementation")
        else:
            self.qt = config.view(-1,1);
            if self.qt.size() != self.config_size:
                raise RuntimeError("New config has inconsistent size compared to previous simulation! Check implementation")

    def step(self, committor_val, onlytst=False):
        with torch.no_grad():
            config_test = torch.zeros_like(self.config)
            self.propose(config_test, committor_val, onlytst)
            committor_val_ = self.committor(config_test)
            self.acceptReject(config_test, committor_val_, onlytst, True)
        self.torch_config = (self.getConfig().flatten()).detach.clone()
        try:
            self.torch.grad.data.zero_()
        except:
            pass

    def step_unbiased(self):
        with torch.no_grad():
            config_test = torch.zeros_like(self.config)
            self.propose(config_test, committor_val, onlytst)
            committor_val_ = self.committor(config_test)
            self.acceptReject(config_test, committor_val_, onlytst, False)
        self.torch_config = (self.getConfig().flatten()).detach.clone()
        try:
            self.torch.grad.data.zero_()
        except:
            pass

    def save(self):


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