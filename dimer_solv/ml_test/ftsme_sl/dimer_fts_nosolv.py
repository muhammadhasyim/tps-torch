import sys
sys.path.insert(0,'/global/home/users/muhammad_hasyim/tps-torch/build/')

import scipy.spatial

#Import necessarry tools from torch
import torch
import torch.distributed as dist
import torch.nn as nn
#Import necessarry tools from tpstorch 
from tpstorch.examples.dimer_solv_ml import MyMLFTSSampler
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
    def __init__(self, react_config, prod_config, num_nodes,boxsize,num_particles):
        super(FTSLayerCustom,self).__init__(react_config, prod_config, num_nodes)
        self.boxsize = boxsize
        self.Np = num_particles
    
    @torch.no_grad()
    def compute_metric(self,x):
        ##(1) Pre-processing so that dimer is at the center
        old_x = x.view(32,3)[:2].clone()

        #Compute the pair distance
        dx = (old_x[0]-old_x[1])
        dx = dx-torch.round(dx/self.boxsize)*self.boxsize
        #Re-compute one of the coordinates and shift to origin
        old_x[0] = dx+old_x[1] 
        x_com = 0.5*(old_x[0]+old_x[1])
       
        #Do the same thing to the string configurations
        new_string = self.string.view(_world_size, self.Np,3).clone()
        #Compute the pair distance
        ds = (new_string[:,0]-new_string[:,1])
        ds = ds-torch.round(ds/self.boxsize)*self.boxsize
	  
        #Re-compute one of the coordinates and shift to origin
        new_string[:,0] = ds+new_string[:,1]
        s_com = 0.5*(new_string[:,0]+new_string[:,1])
        for i in range(self.Np):
            old_x[i] -= x_com
            new_string[:,i] -= s_com
            old_x[i] -= torch.round(old_x[i]/self.boxsize)*self.boxsize 
            new_string[:,i] -= torch.round(new_string[:,i]/self.boxsize)*self.boxsize 
       
        ##(2) Rotate the system using Kabsch algorithm
        dist_sq_list = torch.zeros(_world_size)
        #weights = np.ones(self.Np)/(self.Np-2)
        weights = np.zeros(self.Np)#np.ones(self.Np)/(self.Np-2)
        weights[0] = 1.0
        weights[1] = 1.0
        for i in range(_world_size):
            _, rmsd = scipy.spatial.transform.Rotation.align_vectors(old_x.numpy(), new_string[i].numpy(),weights=weights)
            dist_sq_list[i] = rmsd**2
        return dist_sq_list
   
    #WARNING! Needs to be changed
    #Only testing if configurations are constrained properly
    @torch.no_grad()
    def forward(self,x):
        ##(1) Pre-processing so that dimer is at the center
        old_x = x.view(32,3)[:2].clone()

        #Compute the pair distance
        dx = (old_x[0]-old_x[1])
        dx = dx-torch.round(dx/self.boxsize)*self.boxsize
        #Re-compute one of the coordinates and shift to origin
        old_x[0] = dx+old_x[1] 
        x_com = 0.5*(old_x[0]+old_x[1])
       
        #Do the same thing to the string configurations
        new_string = self.string[_rank].view(self.Np,3).clone()
        #Compute the pair distance
        ds = (new_string[0]-new_string[1])
        ds = ds-torch.round(ds/self.boxsize)*self.boxsize
	  
        #Re-compute one of the coordinates and shift to origin
        new_string[0] = ds+new_string[1]
        s_com = 0.5*(new_string[0]+new_string[1])
        for i in range(self.Np):
            old_x[i] -= x_com
            new_string[i] -= s_com
            old_x[i] -= torch.round(old_x[i]/self.boxsize)*self.boxsize 
            new_string[i] -= torch.round(new_string[i]/self.boxsize)*self.boxsize 
       
        ##(2) Rotate the system using Kabsch algorithm
        dist_sq_list = torch.zeros(_world_size)
        weights = np.zeros(self.Np)
        weights[0] = 1.0
        weights[1] = 1.0
        _, rmsd = scipy.spatial.transform.Rotation.align_vectors(old_x.numpy(), new_string.numpy(),weights=weights)
        return rmsd**2

# This is a sketch of what it should be like, but basically have to also make committor play nicely
# within function
# have omnious committor part that actual committor overrides?
class DimerFTS(MyMLFTSSampler):
    def __init__(self,param,config,rank,beta,mpi_group,ftslayer,output_time, save_config=False):
        super(DimerFTS, self).__init__(param,config.detach().clone(),rank,beta,mpi_group)
        self.output_time = output_time
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
        self.Np = 30+2
        #Configs file Save Alternative, since the default XYZ format is an overkill 
        #self.file = open("newconfigs_{}.txt".format(dist.get_rank()), "w")
    
    @torch.no_grad() 
    def initialize_from_torchconfig(self, config):
        # Don't have to worry about all that much all, can just set it
        self.setConfig(config.detach().clone())
    
    @torch.no_grad() 
    def reset(self):
        self.steps = 0
        self.computeMetric()
        inftscell = self.checkFTSCell(_rank, _world_size)
        if inftscell:
            pass
        else:
            #This teleports the dimer to the origin. But it should be okay?
            state_old = self.getConfig().detach().clone()
            state_old[:2] = self.ftslayer.string[_rank].view(2,3).detach().clone()
            self.setConfig(state_old)
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
        self.torch_config.requires_grad_()
        try:
            self.torch_config.grad.data.zero_()
        except:
            pass
    def step_unbiased(self):
        with torch.no_grad():
            self.stepUnbiased()
        self.torch_config.requires_grad_()
        try:
            self.torch_config.grad.data.zero_()
        except:
            pass

    @torch.no_grad() 
    def isReactant(self, x = None):
        r0 = 2**(1/6.0)
        s = 0.5*r0
        #Compute the pair distance
        if x is None:
            if self.getBondLength() <= r0:
                return True
            else:
                return False
        else:
            if self.getBondLengthConfig(x) <= r0:
                return True
            else:
                return False

    @torch.no_grad() 
    def isProduct(self,x = None):
        r0 = 2**(1/6.0)
        s = 0.5*r0
        if x is None:
            if self.getBondLength() >= r0+2*s:
                return True
            else:
                return False
        else:
            if self.getBondLengthConfig(x) >= r0+2*s:
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
        if self.save_config and self.timestep % self.output_time == 0:
            self.dumpConfig(self.timestep)
