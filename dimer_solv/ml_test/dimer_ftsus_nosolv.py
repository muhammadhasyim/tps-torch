import sys
sys.path.insert(0,'/global/home/users/muhammad_hasyim/tps-torch/build/')

import scipy.spatial
#Import necessarry tools from torch
import torch
import torch.distributed as dist
import torch.nn as nn

#Import necessarry tools from tpstorch 
from tpstorch.examples.dimer_solv_ml import MyMLEXPStringSampler
from tpstorch import _rank, _world_size
import numpy as np
from tpstorch.ml.nn import FTSLayerUS

#Import any other thing
import tqdm, sys

#This function only assumes that the string consists of the dimer without solvent particles
@torch.no_grad()
def dimer_reorient(vec,x,boxsize):
    Np = 2
    ##(1) Pre-processing so that dimer is at the center
    old_x = x.view(2,3).clone()

    #Compute the pair distance
    dx = (old_x[0]-old_x[1])
    dx = dx-torch.round(dx/boxsize)*boxsize
    
    #Re-compute one of the coordinates and shift to origin
    old_x[0] = dx+old_x[1] 
    x_com = 0.5*(old_x[0]+old_x[1])
    
    new_vec = vec.view(Np,3).clone()
    #Compute the pair distance
    ds = (new_vec[0]-new_vec[1])
    ds = ds-torch.round(ds/boxsize)*boxsize
      
    #Re-compute one of the coordinates and shift to origin
    new_vec[0] = ds+new_vec[1]
    s_com = 0.5*(new_vec[0]+new_vec[1])
    for i in range(Np):
        old_x[i] -= x_com
        new_vec[i] -= s_com
        old_x[i] -= torch.round(old_x[i]/boxsize)*boxsize 
        new_vec[i] -= torch.round(new_vec[i]/boxsize)*boxsize 
   
    ##(2) Rotate the system using Kabsch algorithm
    weights = np.ones(Np)
    rotate,rmsd = scipy.spatial.transform.Rotation.align_vectors(new_vec.numpy(),old_x.numpy(), weights=weights)
    for i in range(Np):
        old_x[i] = torch.tensor(rotate.apply(old_x[i].numpy())) 
        old_x[i] -= torch.round(old_x[i]/boxsize)*boxsize 
    return old_x.flatten()

class FTSLayerUSCustom(FTSLayerUS):
    r""" A linear layer, where the paramaters correspond to the string obtained by the 
        general FTS method. Customized to take into account rotational and translational symmetry of the dimer problem
        
        Args:
            react_config (torch.Tensor): starting configuration in the reactant basin. 
            
            prod_config (torch.Tensor): starting configuration in the product basin. 

    """
    def __init__(self, react_config, prod_config, num_nodes,boxsize,kappa_perpend, kappa_parallel, num_particles):
        super(FTSLayerUSCustom,self).__init__(react_config, prod_config, num_nodes, kappa_perpend, kappa_parallel)
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
    
    @torch.no_grad()
    def compute_umbrellaforce(self,x):
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
        rotate, _ = scipy.spatial.transform.Rotation.align_vectors(old_x.numpy(), new_string.numpy(),weights=weights)
        
        force = torch.zeros_like(old_x)
        dX = torch.zeros_like(old_x)
        for i in range(self.Np):
            new_string[i] = torch.tensor(rotate.apply(new_string[i].numpy())) 
            dX[i] = old_x[i]-new_string[i]
            dX[i] -= torch.round(dX[i]/self.boxsize)*self.boxsize
            force[i] += -self.kappa_perpend*dX[i]
        tangent_dx = torch.dot(self.tangent[_rank],dX.flatten())
        force += -(self.kappa_parallel-self.kappa_perpend)*self.tangent[_rank].view(2,3)*tangent_dx
        allforce = torch.zeros(32*3)
        allforce[:6] = force.flatten() 
        return allforce

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
        dist_sq = rmsd**2
        dX = torch.zeros_like(new_x)
        for i in range(self.Np):
            old_x[i] = torch.tensor(rotate.apply(old_x[i].numpy())) 
            dX[i] = old_x[i]-new_string[i]
            dX[i] -= torch.round(dX[i]/self.boxsize)*self.boxsize
        tangent_dx = torch.dot(self.tangent[_rank],dX.flatten())
        return dist_sq+(self.kappa_parallel-self.kappa_perpend)*tangent_dx**2/self.kappa_perpend

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
        #tconfig = ftslayer.string[_rank].view(2,3).detach().clone()
        #tconfig.requires_grad = False
        self.setConfig(config)
        self.Np = 30+2
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
            #Going to set config so that the dimer is that of the string, but rotated in frame
            state_old = self.getConfig().detach().clone()
            #state_old[:2] = self.ftslayer.string[_rank].view(2,3).detach().clone()
            string_old = self.ftslayer.string[_rank].view(2,3).detach().clone()
            distance = torch.abs(string_old[1,2]-string_old[0,2])
            dx = state_old[0]-state_old[1]
            boxsize = self.ftslayer.boxsize
            dx = dx-torch.round(dx/boxsize)*boxsize
            distance_ref = torch.norm(dx)
            dx_norm = dx/distance_ref
            mod_dist = 0.5*(distance_ref-distance)
            state_old[0] = state_old[0]-mod_dist*dx_norm
            state_old[1] = state_old[1]+mod_dist*dx_norm
            state_old[0] -= torch.round(state_old[0]/boxsize)*boxsize 
            state_old[1] -= torch.round(state_old[1]/boxsize)*boxsize 
            self.setConfig(state_old)
    
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
        s = 0.25
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
        s = 0.25
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
