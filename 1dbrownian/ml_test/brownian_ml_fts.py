#Import necessarry tools from torch
import torch
import torch.nn as nn

#Import necessarry tools from tpstorch 
from tpstorch.ml import MLSamplerFTS
from tpstorch.ml.nn import FTSLayer
from tpstorch import dist, _rank, _world_size
from brownian_ml import CommittorNet
import numpy as np

#Import any other thing
import tqdm, sys

#The 1D Brownian particle simulator
class BrownianParticle(MLSamplerFTS):
    def __init__(self, dt, ftslayer, gamma, kT, initial,prefix='',save_config=False):
        super(BrownianParticle, self).__init__(initial.detach().clone())
        
        #Timestep
        self.dt = dt
        self.ftslayer = ftslayer
        #Noise variance
        self.coeff = np.sqrt(2*kT/gamma)
        self.gamma = gamma
        
        #The current position. We make sure that its gradient zero
        self.qt = initial.detach().clone()

        #IO for BP position and committor values
        self.save_config = save_config
        if self.save_config:
            self.qt_io = open("{}_bp_{}.txt".format(prefix,_rank+1),"w")

        #Allocated value for self.qi
        self.invkT = 1/kT
        
        #Tracking steps
        self.timestep = 0

        #Save config size and its flattened version
        self.config_size = initial.size()
        self.flattened_size = initial.flatten().size()
        self.torch_config = (self.qt.flatten()).detach().clone()
        self.torch_config.requires_grad_()

    @torch.no_grad() 
    def initialize_from_torchconfig(self, config):
        #by implementation, torch config cannot be structured: it must be a flat 1D tensor
        #config can ot be flat 
        if config.size() != self.flattened_size:
            raise RuntimeError("Config is not flat! Check implementation")
        else:
            self.qt = config.view(-1,1);
            self.torch_config = config.detach().clone()
            if self.qt.size() != self.config_size:
                raise RuntimeError("New config has inconsistent size compared to previous simulation! Check implementation")
    
    @torch.no_grad() 
    def reset(self):
        self.steps = 0
        self.distance_sq_list = self.ftslayer(self.getConfig().flatten())
        inftscell = self.checkFTSCell(_rank, _world_size)
        if inftscell:
            pass
        else:
            self.setConfig(self.ftslayer.string[_rank])
            pass
    
    def step(self):
        with torch.no_grad():
            config_test = self.qt-4*self.qt*(-1+self.qt**2)*self.dt/self.gamma + self.coeff*torch.normal(torch.tensor([[0.0]]),std=np.sqrt(self.dt))
            self.distance_sq_list = self.ftslayer(config_test.flatten())
            inftscell = self.checkFTSCell(_rank, _world_size)
            if inftscell:
                self.qt = config_test
            else:
                pass
        self.torch_config = (self.getConfig().flatten()).detach().clone()
        self.torch_config.requires_grad_()
        try:
            self.torch.grad.data.zero_()
        except:
            pass
    
    @torch.no_grad() 
    #These functions check whether I'm in the reactant or product basins!
    def isProduct(self, config):
        end = torch.tensor([[1.0]])
        if config.item() >= end.item():
            return True
        else:
            return False

    @torch.no_grad() 
    def isReactant(self, config):
        start = torch.tensor([[-1.0]])
        if config.item() <= start.item():
            return True
        else:
            return False
    
    def step_bc(self):#, basin = None):
        with torch.no_grad():
            #Update one step
            config_test = self.qt-4*self.qt*(-1+self.qt**2)*self.dt/self.gamma + self.coeff*torch.normal(torch.tensor([[0.0]]),std=np.sqrt(self.dt))
            if self.isReactant(config_test.flatten()) or self.isProduct(config_test.flatten()):
                self.qt = config_test.detach().clone()
            else:
                pass
            
        #Don't forget to zero out gradient data after every timestep
        #If you print out torch_config it should be tensor([a,b,c,..,d])
        #where a,b,c and d are some 
        self.torch_config = (self.qt.flatten()).detach().clone()
        self.torch_config.requires_grad_()
        try:
            self.torch.grad.data.zero_()
        except:
            pass
    
    @torch.no_grad() 
    def setConfig(self,config):
        #by implementation, torch config cannot be structured: it must be a flat 1D tensor
        #config can ot be flat 
        self.qt = config.detach().clone()
        self.torch_config = (self.qt.flatten()).detach().clone()
        self.torch_config.requires_grad_()
        try:
            self.torch.grad.data.zero_()
        except:
            pass
    
    @torch.no_grad() 
    def getConfig(self):
        config = (self.qt.flatten()).detach().clone()
        return config

    def save(self):
        #Update timestep counter
        self.timestep += 1
        if self.save_config:
            if self.timestep % 100 == 0:
                self.qt_io.write("{} {}\n".format(self.torch_config[0],1/self.invkT))
                self.qt_io.flush()
    
    def step_unbiased(self):
        #Update one 
        self.qt += -4*self.qt*(-1+self.qt**2)*self.dt/self.gamma + self.coeff*torch.normal(torch.tensor([[0.0]]),std=np.sqrt(self.dt))
        
        #Don't forget to zero out gradient data after every timestep
        self.torch_config = (self.qt.flatten()).detach().clone()
        self.torch_config.requires_grad_()
        try:
            self.torch.grad.data.zero_()
        except:
            pass
