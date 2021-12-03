import sys
sys.path.insert(0,'/global/home/users/muhammad_hasyim/tps-torch/build/')

#Import necessarry tools from torch
import torch
import torch.distributed as dist
import torch.nn as nn

#Import necessarry tools from tpstorch 
from tpstorch.examples.dimer_ml import MyMLFTSSampler
from tpstorch import _rank, _world_size
import numpy as np
from tpstorch.ml.nn import FTSLayer#US

#Import any other thing
import tqdm, sys


class FTSLayerCustom(FTSLayer):
    r""" A linear layer, where the paramaters correspond to the string obtained by the 
        general FTS method. Customized to take into account rotational and translational symmetry of the dimer problem
        
        Args:
            react_config (torch.Tensor): starting configuration in the reactant basin. 
            
            prod_config (torch.Tensor): starting configuration in the product basin. 

    """
    def __init__(self, react_config, prod_config, num_nodes,boxsize):
        super(FTSLayerCustom,self).__init__(react_config, prod_config, num_nodes)
        self.boxsize = boxsize

    @torch.no_grad()
    def compute_metric(self,x):
        #Remove center of mass 
        x = x.view(2,3)
        x_com = 0.5*(x[0]+x[1])
        x[0] -= x_com
        x[1] -= x_com
        
        new_string = self.string.view(_world_size,2,3).clone()
        s_com = (0.5*(new_string[:,0]+new_string[:,1])).detach().clone()#,dim=1)
        new_string[:,0] -= s_com
        new_string[:,1] -= s_com
        
        #Rotate the configuration
        dx = (x[0]-x[1])
        dx = dx-torch.round(dx/self.boxsize)*self.boxsize
        dx /= torch.norm(dx)
        
        ds = (new_string[:,0]-new_string[:,1])
        ds = ds-torch.round(ds/self.boxsize)*self.boxsize
        new_x = torch.zeros_like(new_string)    
        for i in range(_world_size): 
            ds[i] /= torch.norm(ds[i])
            v = torch.cross(dx,ds[i])
            cosine = torch.dot(ds[i],dx)
            new_x[i,0] = x[0] +torch.cross(v,x[0])+torch.cross(v,torch.cross(v,x[0]))/(1+cosine)
            new_x[i,0] -= torch.round(new_x[i,0]/self.boxsize)*self.boxsize
            new_x[i,1] = x[1] +torch.cross(v,x[1])+torch.cross(v,torch.cross(v,x[1]))/(1+cosine)
            new_x[i,1] -= torch.round(new_x[i,1]/self.boxsize)*self.boxsize
        return torch.sum((new_string.view(_world_size,6)-new_x.view(_world_size,6))**2,dim=1)
    
    def forward(self,x):
        #Remove center of mass 
        x = x.view(2,3)
        x_com = 0.5*(x[0]+x[1])
        x[0] -= x_com
        x[1] -= x_com
        
        new_string = self.string[_rank].view(2,3).clone()
        s_com = (0.5*(new_string[0]+new_string[1])).detach().clone()#,dim=1)
        new_string[0] -= s_com
        new_string[1] -= s_com
        
        #Rotate the configuration
        dx = (x[0]-x[1])
        dx = dx-torch.round(dx/self.boxsize)*self.boxsize
        dx /= torch.norm(x[0]-x[1])
        
        ds = (new_string[0]-new_string[1])
        ds = ds-torch.round(ds/self.boxsize)*self.boxsize
        ds /= torch.norm(new_string[0]-new_string[1])
        
        new_x = torch.zeros_like(x)    
        v = torch.cross(dx,ds)
        cosine = torch.dot(ds,dx)
        new_x[0] = x[0] +torch.cross(v,x[0])+torch.cross(v,torch.cross(v,x[0]))/(1+cosine)
        new_x[0] = new_x[0]-torch.round(new_x[0]/self.boxsize)*self.boxsize
        new_x[1] = x[1] +torch.cross(v,x[1])+torch.cross(v,torch.cross(v,x[1]))/(1+cosine)
        new_x[1] = new_x[1]-torch.round(new_x[1]/self.boxsize)*self.boxsize
        return torch.sum((new_string.view(_world_size,6)-new_x.flatten())**2)
# This is a sketch of what it should be like, but basically have to also make committor play nicely
# within function
# have omnious committor part that actual committor overrides?
class DimerFTS(MyMLFTSSampler):
    def __init__(self,param,config,rank,beta,mpi_group,ftslayer,save_config=False):
        super(DimerFTS, self).__init__(param,config.detach().clone(),rank,beta,mpi_group)
        self.save_config = save_config
        self.timestep = 0
        self.torch_config = config
        #Save config size and its flattened version
        self.config_size = config.size()
        self.flattened_size = config.flatten().size()
        
        self.invkT = beta
        self.ftslayer = ftslayer
        #tconfig = ftslayer.string[_rank].view(2,3).detach().clone()
        #tconfig.requires_grad = False
        self.setConfig(config)
        #Configs file Save Alternative, since the default XYZ format is an overkill 
        #self.file = open("newconfigs_{}.txt".format(dist.get_rank()), "w")
    
    @torch.no_grad() 
    def initialize_from_torchconfig(self, config):
        # Don't have to worry about all that much all, can just set it
        self.setConfig(config)
    
    @torch.no_grad() 
    def reset(self):
        self.steps = 0
        self.computeMetric()
        inftscell = self.checkFTSCell(_rank, _world_size)
        if inftscell:
            pass
        else:
            self.setConfig(self.ftslayer.string[_rank].view(2,3).detach().clone())
    def computeMetric(self):
        self.distance_sq_list = self.ftslayer.compute_metric(self.getConfig().flatten())
    
    @torch.no_grad() 
    def step(self):
        state_old = self.getConfig().detach().clone()
        self.stepUnbiased()
        self.computeMetric()
        inftscell = self.checkFTSCell(_rank, _world_size)
        if inftscell:
            pass
        else:
            self.setConfig(state_old)

    def step_unbiased(self):
        with torch.no_grad():
            self.stepUnbiased()
        #self.setConfig(self.getConfig().detach().clone())

    @torch.no_grad() 
    def isProduct(self):
        r0 = 2**(1/6.0)
        s = 0.5*r0
        #Compute the pair distance
        if self.getBondLength() <= r0:
            return True
        else:
            return False

    @torch.no_grad() 
    def isReactant(self):
        r0 = 2**(1/6.0)
        s = 0.5*r0
        if self.getBondLength() >= r0+2*s:
            return True
        else:
            return False
        
    def step_bc(self):
        with torch.no_grad():
            state_old = self.getConfig().detach().clone()
            self.step_unbiased()
            if self.isReactant() or self.isProduct():
                pass
            else:
                #If it's not in the reactant or product state, reset!
                self.setConfig(state_old)

    def save(self):
        self.timestep += 1
        if self.save_config:
            self.dumpConfig(self.timestep)
