import torch
from torch.utils.data import IterableDataset
from itertools import cycle
import tqdm
import numpy as np

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
                    
                    #Run simulation and stop until it falls into the product or reactant state
                    while hitting is False:
                        self.sampler.step_unbiased()
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
