#Import necessarry tools from torch
import torch
import torch.nn as nn

#Import necessarry tools from tpstorch 
from tpstorch.ml import MLSamplerFTS
from tpstorch.ml.nn import FTSLayer
from tpstorch import dist, _rank, _world_size
import numpy as np

#Import any other thing
import tqdm, sys


#The Neural Network for 1D should be very simple.
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
            x = x.view(-1,1)
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

#The 1D Brownian particle simulator
class BrownianParticle(MLSamplerFTS):
    def __init__(self, dt, committor, gamma, kT, initial,prefix='',save_config=False):
        super(BrownianParticle, self).__init__(initial.detach().clone())
        
        #Timestep
        self.dt = dt
        self.committor = committor
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
    
    def step(self):
        with torch.no_grad():
            config_test = self.qt-4*self.qt*(-1+self.qt**2)*self.dt/self.gamma + self.coeff*torch.normal(torch.tensor([[0.0]]),std=np.sqrt(self.dt))
            inftscell = self.checkFTSCell(self.committor(config_test.flatten()), dist.get_rank(), dist.get_world_size())
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
    
    #These functions check whether I'm in the reactant or product basins!
    def isProduct(self, config):
        end = torch.tensor([[1.0]])
        if config.item() >= end.item():
            return True
        else:
            return False

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


"""
class BrownianLoss(CommittorLossEXP):
    def __init__(self, lagrange_bc, world_size, n_boundary_samples, react_configs, prod_configs, mode="random", ref_index=None, batch_size_bc = 0.5):
        super(BrownianLoss,self).__init__()

        #Store the MPI world size (number of MPI processes)
        self.world_size = world_size
        
        # Batch size for BC Loss
        self.batch_size_bc = batch_size_bc
        
        # Stuff for boundary condition loss
        self.react_configs = react_configs #list of reactant basin configurations 
        self.prod_configs = prod_configs #list of product basin configurations
        self.lagrange_bc = lagrange_bc  #strength for quadratic BC loss 
        self.mean_recipnormconst = torch.zeros(1)
        
        #Storing a 1D Tensor of reweighting factors
        self.reweight = [torch.zeros(1) for i in range(_world_size)]
        
        #Choose whether to sample window references randomly or not
        self.mode = mode
        if mode != "random":
            if ref_index is None or ref_index < 0:
                raise ValueError("For non-random choice of window reference, you need to set ref_index!")
            else:
                self.ref_index = torch.tensor(ref_index)
    
    def compute_bc(self, committor):
        #Assume that first dimension is batch dimension
        loss_bc = torch.zeros(1)
        
        #Randomly sample from available BC configurations
        indices_react = torch.randperm(len(self.react_configs))[:int(self.batch_size_bc*len(self.react_configs))]
        indices_prod = torch.randperm(len(self.prod_configs))[:int(self.batch_size_bc*len(self.prod_configs))]
        
        #Computing the BC loss
        react_penalty = torch.sum(committor(self.react_configs[indices_react,:])**2)
        prod_penalty = torch.sum((1.0-committor(self.prod_configs[indices_prod,:]))**2)
        loss_bc += 0.5*self.lagrange_bc*react_penalty#-self.react_lambda*react_penalty
        loss_bc += 0.5*self.lagrange_bc*prod_penalty#-self.prod_lambda*prod_penalty
        return loss_bc/self.world_size

    def computeZl(self,k,fwd_meanwgtfactor,bwrd_meanwgtfactor):
        with torch.no_grad():
            empty = []
            for l in range(_world_size):
                if l > k:
                    empty.append(torch.prod(fwd_meanwgtfactor[k:l]))
                elif l < k:
                    empty.append(torch.prod(bwrd_meanwgtfactor[l:k]))
                else:
                    empty.append(torch.tensor(1.0))
            return torch.tensor(empty).flatten()

    def forward(self, gradients, committor, config, invnormconstants, fwd_weightfactors, bwrd_weightfactors, reciprocal_normconstants):
        self.main_loss = self.compute_loss(gradients, invnormconstants)
        self.bc_loss = self.compute_bc(committor)#, config, invnormconstants)
        
        #renormalize losses
        #get prefactors
        self.reweight = [torch.zeros(1) for i in range(_world_size)]
        fwd_meanwgtfactor = self.reweight.copy()
        dist.all_gather(fwd_meanwgtfactor,torch.mean(fwd_weightfactors))
        fwd_meanwgtfactor = torch.tensor(fwd_meanwgtfactor[:-1])

        bwrd_meanwgtfactor = self.reweight.copy()
        dist.all_gather(bwrd_meanwgtfactor,torch.mean(bwrd_weightfactors))
        bwrd_meanwgtfactor = torch.tensor(bwrd_meanwgtfactor[1:])
        
        #Randomly select a window as a free energy reference and broadcast that index across all processes
        if self.mode == "random":
            self.ref_index = torch.randint(low=0,high=_world_size,size=(1,))
            dist.broadcast(self.ref_index, src=0)
        
        #Computing the reweighting factors, z_l in  our notation
        self.reweight = self.computeZl(self.ref_index,fwd_meanwgtfactor,bwrd_meanwgtfactor)
        self.reweight.div_(torch.sum(self.reweight))  #normalize
        
        #Use it first to compute the mean inverse normalizing constant
        mean_recipnormconst = torch.mean(invnormconstants)
        mean_recipnormconst.mul_(self.reweight[_rank])

        #All reduce the mean invnormalizing constant
        dist.all_reduce(mean_recipnormconst)

        #renormalize main_loss
        self.main_loss.mul_(self.reweight[_rank])
        dist.all_reduce(self.main_loss)
        self.main_loss.div_(mean_recipnormconst)

        #normalize bc_loss
        dist.all_reduce(self.bc_loss)
        return self.main_loss+self.bc_loss
"""
