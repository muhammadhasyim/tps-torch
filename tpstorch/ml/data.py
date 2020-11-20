import torch
from torch.utils.data import IterableDataset
from itertools import cycle

class EXPReweightSimulation(IterableDataset):
    
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
                self.sampler.step(self.out)
                
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
                        self.sampler.forward_weightfactor, #this is the un-normalized weighting factor, this should compute w_{l+1}/w_{l} where l is the l-th window
                        self.sampler.backward_weightfactor, #this is the un-normalized weighting factor, this should compute w_{l-1}/w_{l} where l is the l-th window
                        ) 
                 
    def __iter__(self):
        #Cycle through every period indefinitely
        return cycle(self.runSimulation())
