import sys
sys.path.insert(0,'/global/home/users/muhammad_hasyim/tps-torch/build/')

#Import necessarry tools from torch
import torch
import torch.distributed as dist
import torch.nn as nn

#Import necessarry tools from tpstorch 
from tpstorch.examples.dimer_ml import MyMLEXPStringSampler
from tpstorch import _rank, _world_size
import numpy as np
from tpstorch.ml.nn import FTSLayerUS

#Import any other thing
import tqdm, sys


class FTSLayerUSCustom(FTSLayerUS):
    r""" A linear layer, where the paramaters correspond to the string obtained by the 
        general FTS method. Customized to take into account rotational and translational symmetry of the dimer problem
        
        Args:
            react_config (torch.Tensor): starting configuration in the reactant basin. 
            
            prod_config (torch.Tensor): starting configuration in the product basin. 

    """
    def __init__(self, react_config, prod_config, num_nodes,boxsize,kappa_perpend, kappa_parallel):
        super(FTSLayerUSCustom,self).__init__(react_config, prod_config, num_nodes, kappa_perpend, kappa_parallel)
        self.boxsize = boxsize

    @torch.no_grad()
    def compute_metric(self,x):
        ##(1) Remove center of mass 
        old_x = x.view(2,3).detach().clone()

        #Compute the pair distance
        dx = (old_x[0]-old_x[1])
        dx = dx-torch.round(dx/self.boxsize)*self.boxsize
        
        #Re-compute one of the coordinates and shift to origin
        old_x[0] = dx+old_x[1] 
        x_com = 0.5*(old_x[0]+old_x[1])
        old_x[0] -= x_com
        old_x[1] -= x_com
        
        new_string = self.string.view(_world_size,2,3).detach().clone()
        
        #Compute the pair distance
        ds = (new_string[:,0]-new_string[:,1])
        ds = ds-torch.round(ds/self.boxsize)*self.boxsize
        
        #Re-compute one of the coordinates and shift to origin
        new_string[:,0] = ds+new_string[:,1]
        s_com = 0.5*(new_string[:,0]+new_string[:,1])#.detach().clone()#,dim=1)
        new_string[:,0] -= s_com
        new_string[:,1] -= s_com
        
        ##(2) Rotate the configuration
        dx /= torch.norm(dx)
        new_x = torch.zeros_like(new_string)    
        for i in range(_world_size): 
            ds[i] /= torch.norm(ds[i])
            v = torch.cross(dx,ds[i])
            cosine = torch.dot(ds[i],dx)
            if cosine < 0:
                new_string[i] *= -1
                ds[i] *= -1
                v *= -1
                cosine = torch.dot(ds[i],dx)
            new_x[i,0] = old_x[0] +torch.cross(v,old_x[0])+torch.cross(v,torch.cross(v,old_x[0]))/(1+cosine)
            new_x[i,1] = old_x[1] +torch.cross(v,old_x[1])+torch.cross(v,torch.cross(v,old_x[1]))/(1+cosine)
        return torch.sum((new_string.view(_world_size,6)-new_x.view(_world_size,6))**2,dim=1)
    
    @torch.no_grad()
    def compute_umbrellaforce(self,x):
        #For now, don't rotate or translate the system
        ##(1) Remove center of mass 
        new_x = x.view(2,3).detach().clone()
        new_string = self.string[_rank].view(2,3).detach().clone()
        
        #Compute the pair distance
        dx = (new_x[0]-new_x[1])
        dx = dx-torch.round(dx/self.boxsize)*self.boxsize
        
        #Re-compute one of the coordinates and shift to origin
        new_x[0] = dx+new_x[1] 
        x_com = 0.5*(new_x[0]+new_x[1])
        new_x[0] -= x_com
        new_x[1] -= x_com
        
        new_string = self.string[_rank].view(2,3).detach().clone()
        
        #Compute the pair distance
        ds = (new_string[0]-new_string[1])
        ds = ds-torch.round(ds/self.boxsize)*self.boxsize
        
        
        #Re-compute one of the coordinates and shift to origin
        new_string[0] = ds+new_string[1]
        s_com = 0.5*(new_string[0]+new_string[1])#.detach().clone()#,dim=1)
        new_string[0] -= s_com
        new_string[1] -= s_com
        
        ##(2) Rotate the configuration
        dx /= torch.norm(dx)
        ds /= torch.norm(ds)
        #v = torch.cross(dx,ds)
        v = torch.cross(ds,dx)
        cosine = torch.dot(ds,dx)
        if cosine < 0:
            ds *= -1
            new_string *= -1
            v *= -1
            cosine = torch.dot(ds,dx)
        #new_x[0] += torch.cross(v,new_x[0])+torch.cross(v,torch.cross(v,new_x[0]))/(1+cosine)
        #new_x[1] += torch.cross(v,new_x[1])+torch.cross(v,torch.cross(v,new_x[1]))/(1+cosine)
        new_string[0] += torch.cross(v,new_string[0])+torch.cross(v,torch.cross(v,new_string[0]))/(1+cosine)
        new_string[1] += torch.cross(v,new_string[1])+torch.cross(v,torch.cross(v,new_string[1]))/(1+cosine)
        dX = new_x.flatten()-new_string.flatten()
        dX = dX-torch.round(dX/self.boxsize)*self.boxsize
        tangent_dx = torch.dot(self.tangent[_rank],dX)
        return -self.kappa_perpend*dX-(self.kappa_parallel-self.kappa_perpend)*self.tangent[_rank]*tangent_dx
    
    def forward(self,x):
        ##(1) Remove center of mass 
        new_x = x.view(2,3).detach().clone()
        #new_string = self.string[_rank].view(2,3).detach().clone()
        #Compute the pair distance
        dx = (new_x[0]-new_x[1])
        dx = dx-torch.round(dx/self.boxsize)*self.boxsize
        
        #Re-compute one of the coordinates and shift to origin
        new_x[0] = dx+new_x[1] 
        x_com = 0.5*(new_x[0]+new_x[1])
        new_x[0] -= x_com
        new_x[1] -= x_com
        
        new_string = self.string[_rank].view(2,3).detach().clone()
        
        #Compute the pair distance
        ds = (new_string[0]-new_string[1])
        ds = ds-torch.round(ds/self.boxsize)*self.boxsize
        
        #Re-compute one of the coordinates and shift to origin
        new_string[0] = ds+new_string[1]
        s_com = 0.5*(new_string[0]+new_string[1])#.detach().clone()#,dim=1)
        new_string[0] -= s_com
        new_string[1] -= s_com
        
        ##(2) Rotate the configuration
        dx /= torch.norm(dx)
        ds /= torch.norm(ds)
        v = torch.cross(ds,dx)
        #v = torch.cross(dx,ds)
        cosine = torch.dot(ds,dx)
        if cosine < 0:
            ds *= -1
            new_string *= -1
            v *= -1
            cosine = torch.dot(ds,dx)
        #new_x[0] += torch.cross(v,new_x[0])+torch.cross(v,torch.cross(v,new_x[0]))/(1+cosine)
        #new_x[1] += torch.cross(v,new_x[1])+torch.cross(v,torch.cross(v,new_x[1]))/(1+cosine)
        new_string[0] += torch.cross(v,new_string[0])+torch.cross(v,torch.cross(v,new_string[0]))/(1+cosine)
        new_string[1] += torch.cross(v,new_string[1])+torch.cross(v,torch.cross(v,new_string[1]))/(1+cosine)
        dX = new_x.flatten()-new_string.flatten()
        dX = dX-torch.round(dX/self.boxsize)*self.boxsize
        dist_sq = torch.sum(dX**2)
        tangent_dx = torch.sum(self.tangent[_rank]*dX)
        return dist_sq+(self.kappa_parallel-self.kappa_perpend)*tangent_dx**2/self.kappa_perpend#torch.sum((tangent*(self.string-x))**2,dim=1)

# This is a sketch of what it should be like, but basically have to also make committor play nicely
# within function
# have omnious committor part that actual committor overrides?
class DimerFTSUS(MyMLEXPStringSampler):
    def __init__(self,param,config,rank,beta,kappa, mpi_group,ftslayer,output_time, save_config=False):
        super(DimerFTSUS, self).__init__(param,config.detach().clone(),rank,beta,kappa, mpi_group)
        self.output_time = output_time
        self.save_config = save_config
        self.timestep = 0
        self.torch_config = config
        #Save config size and its flattened version
        self.config_size = config.size()
        self.flattened_size = config.flatten().size()
        
        self.invkT = beta
        self.ftslayer = ftslayer
        self.setConfig(config)
    
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
            self.setConfig(self.ftslayer.string[_rank].view(2,3).detach().clone())
    
    def computeMetric(self):
        self.distance_sq_list = self.ftslayer.compute_metric(self.getConfig().flatten())
    
    def computeWForce(self,x):
        return self.ftslayer.compute_umbrellaforce(x)
    
    @torch.no_grad() 
    def step(self):
        state_old = self.getConfig().detach().clone()
        self.stepBiased(self.computeWForce(state_old.flatten()))
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
    @torch.no_grad() 
    def step_bc(self):
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
