#Import necessarry tools from torch
import torch
import torch.distributed as dist
import torch.nn as nn

#Import necessarry tools from tpstorch 
from tpstorch.ml import MLSamplerFTS
from tpstorch.ml.nn import FTSLayer
from tpstorch.examples.mullerbrown_ml import MyMLFTSSampler
import numpy as np

#Import any other thing
import tqdm, sys


#A single hidden layer NN, where some nodes possess the string configurations
class CommittorFTSNet(nn.Module):
    def __init__(self, d, num_nodes, start, end, fts_size, unit=torch.relu):
        super(CommittorFTSNet, self).__init__()
        self.num_nodes = num_nodes
        self.d = d
        self.unit = unit
        self.lin1 = FTSLayer(start, end, fts_size)
        self.lin3 = nn.Linear(d, num_nodes-fts_size+1, bias=True)
        self.lin2 = nn.Linear(num_nodes, 1, bias=True)
        self.broadcast()

    def forward(self, x):
        #At the moment, x is flat. So if you want it to be 2x1 or 3x4 arrays, then you do it here!
        if x.shape == torch.Size([self.d]):
            x = x.view(-1,2)
        x1 = self.lin1(x)
        x1 = self.unit(x1)
        x3 = self.lin3(x)
        x3 = self.unit(x3)
        x = torch.cat((x1,x3),dim=1)
        x = self.lin2(x)
        #x = self.lin2(x)
        return torch.sigmoid(x)
    
    def broadcast(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                dist.broadcast(param.data,src=0)

# This is a sketch of what it should be like, but basically have to also make committor play nicely
# within function
# have omnious committor part that actual committor overrides?
class MullerBrown(MyMLFTSSampler):
    def __init__(self,param,config,rank,dump,beta,mpi_group,committor,save_config=False):
        super(MullerBrown, self).__init__(param,config.detach().clone(),rank,dump,beta,mpi_group)

        self.committor = committor

        self.save_config = save_config
        self.timestep = 0

        #Save config size and its flattened version
        self.config_size = config.size()
        self.flattened_size = config.flatten().size()
        self.torch_config = config
        
        #Configs file Save Alternative, since the default XYZ format is an overkill 
        self.file = open("configs_{}.txt".format(dist.get_rank()), "w")
    
    def initialize_from_torchconfig(self, config):
        # Don't have to worry about all that much all, can just set it
        self.setConfig(config)

    def step(self):
        with torch.no_grad():
            config_test = torch.zeros_like(self.torch_config)
            self.propose(config_test, 0.0, False)
            committor_val_ = self.committor(config_test)
            inftscell = self.checkFTSCell(self.committor(config_test.flatten()), dist.get_rank(), dist.get_world_size())
            if inftscell:
                self.acceptReject(config_test)#, committor_val_.item(), False, True)
            else:
                pass
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
            self.acceptReject(config_test)#, committor_val_.item(), False, True)
        self.torch_config = (self.getConfig().flatten()).detach().clone()
        self.torch_config.requires_grad_()
        try:
            self.torch.grad.data.zero_()
        except:
            pass
    
    def isProduct(self,config):
        end = torch.tensor([[0.6,0.08]])
        radii = 0.025
        end_ = config-end
        end_ = end_.pow(2).sum()**0.5
        if end_ <= radii:
            return True
        else:
            return False

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
                self.acceptReject(config_test)
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
            if self.timestep % 10 == 0:
         #       self.dumpConfig(0)
                self.file.write("{} {} \n".format(self.torch_config[0].item(), self.torch_config[1].item()))
                self.file.flush()
