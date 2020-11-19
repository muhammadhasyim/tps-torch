import torch
from torch.utils.data import IterableDataset
from itertools import cycle

class SimulationDataset(IterableDataset):
    
    def __init__(self, sampler, committor, period):
        #The MD/MC Simulation sampling data
        self.sampler = sampler
        self.period = period

        self.continue_simulation = True
        
        #The committor, which we will continuously call for gradient computations
        #We also run the first backward pass here,
        self.committor  = committor
        
        #Zero out any gradients
        self.committor.zero_grad()

        #Forward pass
        self.out = self.committor(self.sampler.torch_config)
        
        #Compute the new set of reweighting factors from this new step 
        self.sampler.computeFactors(self.out)
        
        #Backprop to compute gradients of x
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
                yield ( self.sampler.torch_config,
                        torch.autograd.grad(self.committor(self.sampler.torch_config), self.sampler.torch_config, create_graph=True)[0], #The gradient of commmittor with respect to input
                        self.sampler.invnormconstant, #Inverse of normalizing constant, denoted as c(x) in the manuscript
                        self.sampler.weightfactor, #this is the un-normalized weighting factor
                        ) 
                 
    def __iter__(self):
        return cycle(self.runSimulation())
