#Import necessarry tools from torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn as nn

#Import necessarry tools from tpstorch 
from tpstorch.ml import MLSamplerEXP
from mullerbrown_ml import MySampler
import numpy as np

#Import any other thing
import tqdm, sys

#dist.init_process_group(backend='mpi')

class CommittorNet(nn.Module):
    def __init__(self, d, num_nodes, unit=torch.relu):
        super(CommittorNet, self).__init__()
        self.num_nodes = num_nodes
        self.d = d
        self.unit = unit
        self.lin1 = nn.Linear(d, num_nodes, bias=True)
        self.lin3 = nn.Linear(num_nodes, num_nodes, bias=True)
        self.lin2 = nn.Linear(num_nodes, 1, bias=False)
        self.broadcast()

    def forward(self, x):
        x = self.lin1(x)
        x = self.unit(x)
        x = self.lin3(x)
        x = self.unit(x)
        x = self.lin2(x)
        return torch.sigmoid(x)
    
    def broadcast(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                dist.broadcast(param.data,src=0)

class MullerBrown(MySampler):
    def __init__(self,param,config,rank,dump,beta,kappa,mpi_group,committor,save_config=False):
        super(MullerBrown, self).__init__(param,config.detach().clone(),rank,dump,beta,kappa,mpi_group)
        #super(MullerBrown, self).__init__(config.detach().clone())
        #super(MullerBrown, self).__init__(param,config.detach().clone(),rank,dump,beta,kappa,mpi_group)

        self.committor = committor

        self.save_config = save_config
        self.timestep = 0

        #Save config size and its flattened version
        self.config_size = config.size()
        self.flattened_size = config.flatten().size()
        self.torch_config = config

    @torch.no_grad() 
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

    @torch.no_grad() 
    def isProduct(self,config):
        end = torch.tensor([[0.6,0.08]])
        radii = 0.025
        end_ = config-end
        end_ = end_.pow(2).sum()**0.5
        if end_ <= radii:
            return True
        else:
            return False

    @torch.no_grad() 
    def isReactant(self,config):
        start = torch.tensor([[-0.5,1.5]])
        radii = 0.025
        start_ = config-start
        start_ = start_.pow(2).sum()**0.5
        if start_ <= radii:
            return True
        else:
            return False

    def step_bc(self):
        with torch.no_grad():
            config_test = torch.zeros_like(self.torch_config)
            self.propose(config_test, 0.0, False)
            if self.isReactant(config_test.flatten()) or self.isProduct(config_test.flatten()):
                self.acceptReject(config_test, 0.0, False, False)
            else:
                pass
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
