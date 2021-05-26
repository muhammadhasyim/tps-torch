"""
A place to put classes and method which we will discard for all future implementations. We'll still include in case we want to revert back
"""

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.utils.data import IterableDataset
import torch.distributed as dist
import torch.optim.functional as F
from itertools import cycle
import tqdm
import numpy as np

class FTSLayer(nn.Module):
    r""" A linear layer, where the paramaters correspond to the string obtained by the 
        general FTS method. 
        
        Args:
            react_config (torch.Tensor): starting configuration in the reactant basin. 
            
            prod_config (torch.Tensor): starting configuration in the product basin. 

    """
    def __init__(self, react_config, prod_config, num_nodes):
        super().__init__()
            
        #Declare my string as NN paramaters and disable gradient computations
        string = torch.vstack([(1-s)*react_config+s*prod_config for s in np.linspace(0, 1, num_nodes)])
        self.string = nn.Parameter(string) 
        self.string.requires_grad = False 
    
    def forward(self, x):
        #The weights of this layer models hyperplanes wedged between each node
        w_times_x= torch.matmul(x,(self.string[1:]-self.string[:-1]).t())
        
        #The bias so that at the half-way point between two strings, the function is zero
        bias = -torch.sum(0.5*(self.string[1:]+self.string[:-1])*(self.string[1:]-self.string[:-1]),dim=1)
        
        return torch.add(w_times_x, bias)



class EXPReweightSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    
    This implementation has an additional step which is reweighting the computed 
    gradients through free-energy estimation techniques. Currently only implementng
    exponential averaging (EXP) because it is cheap. 
    
    Any more detailed implementation should be consulted on torch.optim.SGD
    """

    def __init__(self, params, sampler=required, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, mode="random", ref_index=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(EXPReweightSGD, self).__init__(params, defaults)
       
        #Storing a 1D Tensor of reweighting factors
        self.reweight = [torch.zeros(1) for i in range(dist.get_world_size())]
        #Choose whether to sample window references randomly or not
        self.mode = mode
        if mode != "random":
            if ref_index is None or ref_index < 0:
                raise ValueError("For non-random choice of window reference, you need to set ref_index!")
            else:
                self.ref_index = torch.tensor(ref_index)
    def __setstate__(self, state):
        super(EXPReweightSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None, fwd_weightfactors=required, bwrd_weightfactors=required, reciprocal_normconstants=required):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        #Average out the batch of weighting factors (unique to each process)
        #and distribute them across all processes.
        #TO DO: combine weight factors into a single array so that we have one contigous memory to distribute
        self.reweight = [torch.zeros(1) for i in range(dist.get_world_size())]
        fwd_meanwgtfactor = self.reweight.copy()
        dist.all_gather(fwd_meanwgtfactor,torch.mean(fwd_weightfactors))
        fwd_meanwgtfactor = torch.tensor(fwd_meanwgtfactor[:-1])

        bwrd_meanwgtfactor = self.reweight.copy()
        dist.all_gather(bwrd_meanwgtfactor,torch.mean(bwrd_weightfactors))
        bwrd_meanwgtfactor = torch.tensor(bwrd_meanwgtfactor[1:])
        
        #Randomly select a window as a free energy reference and broadcast that index across all processes
        if self.mode == "random":
            self.ref_index = torch.randint(low=0,high=dist.get_world_size(),size=(1,))
            dist.broadcast(self.ref_index, src=0)
        
        #Computing the reweighting factors, z_l in  our notation
        self.reweight = computeZl(self.ref_index.item(),fwd_meanwgtfactor,bwrd_meanwgtfactor)#newcontainer)
        self.reweight.div_(torch.sum(self.reweight))  #normalize
        
        #Use it first to compute the mean inverse normalizing constant
        mean_recipnormconst = torch.mean(reciprocal_normconstants)#invnormconstants)
        mean_recipnormconst.mul_(self.reweight[dist.get_rank()])

        #All reduce the mean invnormalizing constant
        dist.all_reduce(mean_recipnormconst)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                #Gradient of parameters
                #p.grad should be the average of grad(x)/c(x) over the minibatch
                d_p = p.grad

                #Multiply with the window's respective reweighting factor
                d_p.mul_(self.reweight[dist.get_rank()])
                
                #All reduce the gradients
                dist.all_reduce(d_p)
                
                #Divide in-place by the mean inverse normalizing constant
                d_p.div_(mean_recipnormconst)
                
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                p.add_(d_p, alpha=-group['lr'])
        
        return mean_recipnormconst, self.reweight

class OldEXPReweightSimulation(IterableDataset):
    
    def __init__(self, sampler, committor, period):
        ## Store the MD/MC Simulator, which samples our data
        self.sampler = sampler
        ## The number of timesteps we do per iteration of the optimization
        self.period = period

        ## A flag which is set False when we detect something wrong
        self.continue_simulation = True
        
        ## The committor, which we will continuously call for gradient computations
        self.committor  = committor
        
        ## We also run the first forward-backward pass here, using the torch_config 
        ## saved in our sampler
        #Zero out any gradients
        self.committor.zero_grad()

        #Forward pass
        self.out = self.committor(self.sampler.torch_config)
        
        #Compute the first set of reweighting factors from our initial condition
        self.sampler.computeFactors(self.out)
        
        #Backprop to compute gradients w.r.t. x
        self.out.backward()
    
    def runSimulation(self):
        while self.continue_simulation:
            for i in range(self.period):
                #Take one step
                self.sampler.step(self.out,onlytst=False)
                
                #Save config
                self.sampler.save()        
                
                #Zero out any gradients
                self.committor.zero_grad()

                #Forward pass
                self.out = self.committor(self.sampler.torch_config)
                
                #Compute the new set of reweighting factors from this new step 
                self.sampler.computeFactors(self.out)
                
                #Backprop to compute gradients of x
                self.out.backward()
                 
            if torch.sum(torch.isnan(self.out)) > 0:
                raise ValueError("Committor value is NaN!")
            else:
                yield ( self.sampler.torch_config, #The configuration of the system 
                        torch.autograd.grad(self.committor(self.sampler.torch_config), self.sampler.torch_config, create_graph=True)[0], #The gradient of commmittor with respect to input
                        self.sampler.reciprocal_normconstant, #Inverse of normalizing constant, denoted as 1/c(x) in the manuscript
                        self.sampler.fwd_weightfactor, #this is the un-normalized weighting factor, this should compute w_{l+1}/w_{l} where l is the l-th window
                        self.sampler.bwrd_weightfactor, #this is the un-normalized weighting factor, this should compute w_{l-1}/w_{l} where l is the l-th window
                        ) 
                 
    def __iter__(self):
        #Cycle through every period indefinitely
        return cycle(self.runSimulation())

## TO DO: Revamp how we do committor analysis!
class TSTValidation(IterableDataset):
    
    def __init__(self, sampler, committor, period):
        ## Store the MD/MC Simulator, which samples our data
        self.sampler = sampler

        ## The number of timesteps we do per iteration of the optimization
        self.period = period

        ## A flag which is set False when we detect something wrong
        self.continue_simulation = True
        
        ## The committor, which we will continuously call for gradient computations
        self.committor  = committor
        
        ## We also run the first forward-backward pass here, using the torch_config 
        ## saved in our sampler
        
        #Zero out any gradients
        self.committor.zero_grad()

        #Forward pass
        self.out = self.committor(self.sampler.torch_config)
        
        #Backprop to compute gradients w.r.t. x
        self.out.backward()
    
    def generateInitialConfigs(self):
        while self.continue_simulation:
            for i in range(self.period):

                #Take one step in the transition state region
                self.sampler.step(self.out,onlytst=True)
                
                #Zero out any gradients
                self.committor.zero_grad()

                #Forward pass
                self.out = self.committor(self.sampler.torch_config)
                
                #Backprop to compute gradients of x
                self.out.backward()
                 
            if torch.sum(torch.isnan(self.out)) > 0:
                raise ValueError("Committor value is NaN!")
            else:
                yield ( self.sampler.torch_config, #The configuration of the system 
                        self.committor(self.sampler.torch_config), #the commmittor value at that point, which we will compare it with. 
                        ) 
    
    #Validation for-loop, performing committor analysis 
    def validate(self, batch, trials=10, validation_io=None, product_checker= None, reactant_checker = None):
        
        #Separate out batch data from configurations and the neural net's prediction
        configs, committor_values = batch
        
        if product_checker is None or reactant_checker is None:
            raise RuntimeError("User must supply a function that checks if a configuration is in the product state or not!")
        else:
            #Use tqdm to track the progress
            for idx, initial_config in tqdm.tqdm(enumerate(configs)):
                counts = []
                for i in range(trials):
                    hitting = False

                    #Override the current configuration using a fixed initialization routine 
                    self.sampler.initialize_from_torchconfig(initial_config.detach().clone())
                    #Zero out any gradients
                    self.committor.zero_grad()
                    #Forward pass
                    self.out = self.committor(self.sampler.torch_config)
                    #Backprop to compute gradients of x
                    self.out.backward()
                    
                    #Run simulation and stop until it falls into the product or reactant state
                    while hitting is False:
                        self.sampler.step_unbiased()
                        #Zero out any gradients
                        self.committor.zero_grad()
                        #Forward pass
                        self.out = self.committor(self.sampler.torch_config)
                        #Backprop to compute gradients of x
                        self.out.backward()
                        
                        if product_checker(self.sampler.torch_config) is True:
                            counts.append(1.0)
                            hitting = True
                        if reactant_checker(self.sampler.torch_config) is True:
                            counts.append(0.0)
                            hitting = True
                
                #Compute the committor after a certain number of trials
                counts = np.array(counts)
                mean_count = np.mean(counts) 
                conf_count = 1.96*np.std(counts)/len(counts)**0.5 #do 95 % confidence interval
                
                #Save into io
                if validation_io is not None:
                    validation_io.write('{} {} {} \n'.format(committor_values[idx].item(),mean_count, conf_count))
                    validation_io.flush()
                 
    def __iter__(self):
        #Cycle through every period indefinitely
        return cycle(self.generateInitialConfigs())
