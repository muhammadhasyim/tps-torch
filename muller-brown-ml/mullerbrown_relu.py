#Import necessarry tools from torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn as nn

#Import necessarry tools from tpstorch 
from tpstorch.ml.optim import project_simplex
from tpstorch.ml import MLSamplerEXP
from tpstorch.ml.nn import CommittorLossEXP
from mullerbrown_ml import MySampler
import numpy as np

#Import any other thing
import tqdm, sys

class CommittorNet(nn.Module):
    def __init__(self, d, num_nodes, unit=torch.relu):
        super(CommittorNet, self).__init__()
        self.num_nodes = num_nodes
        self.d = d
        self.unit = unit
        self.lin1 = nn.Linear(d, num_nodes, bias=True)
        self.lin2 = nn.Linear(num_nodes, 1, bias=False)
        self.thresh = torch.sigmoid
        self.project()
        self.broadcast()

    def forward(self, x):
        #At the moment, x is flat. So if you want it to be 2x1 or 3x4 arrays, then you do it here!
        x = self.lin1(x)
        x = self.unit(x)
        x = self.lin2(x)
        x = self.thresh(x)
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
    def __init__(self,param,config,rank,dump,beta,kappa,mpi_group,committor,save_config=False):
        super(MullerBrown, self).__init__(param,config.detach().clone(),rank,dump,beta,kappa,mpi_group)

        self.committor = committor

        self.save_config = save_config
        self.timestep = 0

        #Save config size and its flattened version
        self.config_size = config.size()
        self.flattened_size = config.flatten().size()
        self.torch_config = config

    def initialize_from_torchconfig(self, config):
        # Don't have to worry about all that much all, can just set it
        self.setConfig(config)

    def step(self, committor_val, onlytst=False):
        with torch.no_grad():
            config_test = torch.zeros_like(self.torch_config)
            self.propose(config_test, committor_val, onlytst)
            committor_val_ = self.committor(config_test)
            self.acceptReject(config_test, committor_val_, onlytst, True)
        self.torch_config = (self.getConfig().flatten()).detach().clone()
        self.torch_config.requires_grad_()
        try:
            self.torch.grad.data.zero_()
        except:
            pass

    def step_unbiased(self):
        with torch.no_grad():
            config_test = torch.zeros_like(self.torch_config)
            self.propose(config_test, 0.0, False)
            self.acceptReject(config_test, 0.0, False, False)
        self.torch_config = (self.getConfig().flatten()).detach().clone()
        self.torch_config.requires_grad_()
        try:
            self.torch.grad.data.zero_()
        except:
            pass

    def save(self):
        self.timestep += 1
        if self.save_config:
            if self.timestep % 100 == 0:
                self.dumpConfig(0)


class MullerBrownLoss(CommittorLossEXP):
    def __init__(self, lagrange_bc, batch_size,start,end,radii,world_size,n_boundary_samples,react_configs,prod_configs):
        super(MullerBrownLoss,self).__init__()
        self.lagrange_bc = lagrange_bc
        self.start = start
        self.end = end
        self.radii = radii
        self.world_size = world_size
        self.react_configs = react_configs
        self.prod_configs = prod_configs
        self.react_lambda = torch.zeros(1)
        self.prod_lambda = torch.zeros(1)

    def compute_bc(self, committor, configs, invnormconstants):
        #Assume that first dimension is batch dimension
        loss_bc = torch.zeros(1)
        if dist.get_rank() == 0:
            print(self.react_configs)
            print(committor(self.react_configs))
            print(torch.mean(committor(self.react_configs)))
            print(self.prod_configs)
            print(committor(self.prod_configs))
            print(torch.mean(committor(self.prod_configs)))
        self.react_lambda = self.react_lambda.detach()
        self.prod_lambda = self.prod_lambda.detach()
        react_penalty = torch.mean(committor(self.react_configs))
        prod_penalty = torch.mean(1.0-committor(self.prod_configs))
        #self.react_lambda -= self.lagrange_bc*react_penalty
        #self.prod_lambda -= self.lagrange_bc*prod_penalty
        loss_bc += 0.5*self.lagrange_bc*react_penalty**2-self.react_lambda*react_penalty
        loss_bc += 0.5*self.lagrange_bc*prod_penalty**2-self.prod_lambda*prod_penalty
        return loss_bc/self.world_size
