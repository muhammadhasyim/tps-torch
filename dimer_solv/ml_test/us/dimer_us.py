import sys
sys.path.insert(0,'/global/home/users/muhammad_hasyim/tps-torch/build/')

#Import necessarry tools from torch
import torch
import torch.distributed as dist
import torch.nn as nn

#Import necessarry tools from tpstorch 
from tpstorch.examples.dimer_solv_ml import MyMLEXPSampler
from tpstorch import _rank, _world_size
import numpy as np
from tpstorch.ml.nn import FTSLayerUS

#Import any other thing
import tqdm, sys


# This is a sketch of what it should be like, but basically have to also make committor play nicely
# within function
# have omnious committor part that actual committor overrides?
class DimerUS(MyMLEXPSampler):
    def __init__(self,param,config,rank,beta,kappa, mpi_group,output_time, save_config=False):
        super(DimerUS, self).__init__(param,config.detach().clone(),rank,beta,kappa, mpi_group)
        self.output_time = output_time
        self.save_config = save_config
        self.timestep = 0
        self.torch_config = config
        #Save config size and its flattened version
        self.config_size = config.size()
        self.flattened_size = config.flatten().size()
        
        self.invkT = beta
        self.setConfig(config)
        self.Np = 30+2
        
        self.dt =0
        self.gamma = 0
        #Read the local param file to get info on step size and friction constant
        with open("param","r") as f:
            for line in f:
                test = line.strip()
                test = test.split()
                if (len(test) == 0):
                    continue
                else:
                    if test[0] == "gamma":
                        self.gamma = float(test[1])
                    elif test[0] == "dt":
                        self.dt = float(test[1])
    
    @torch.no_grad() 
    def initialize_from_torchconfig(self, config):
        # Don't have to worry about all that much all, can just set it
        self.setConfig(config.detach().clone())
    
    def computeWForce(self, committor_val, qval):
        return -self.kappa*self.torch_config.grad.data*(committor_val-qval)#/self.gamma 
    
    def step(self, committor_val, onlytst=False):
        with torch.no_grad():
            #state_old = self.getConfig().detach().clone()
            if onlytst:
                self.stepBiased(self.computeWForce(committor_val, 0.5))#state_old.flatten()))
            else:
                self.stepBiased(self.computeWForce(committor_val, self.qvals[_rank]))#state_old.flatten()))
        self.torch_config.requires_grad_()
        self.torch_config.grad.data.zero_()

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
