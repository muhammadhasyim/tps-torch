import torch
import numpy as np

class FTSSimulation:
    def __init__(self, sampler, committor, period, batch_size, dimN):
        
        ## Store the MD/MC Simulator, which samples our data
        self.sampler = sampler
        
        ## The number of timesteps we do per iteration of the optimization
        self.period = period
        
        ## The number of iterations we do per optimization step
        self.batch_size = batch_size
        
        ## The size of the problem
        self.dimN = dimN

        ## A flag which is set False when we detect something wrong
        self.continue_simulation = True
        
        ## The committor, which we will continuously call for gradient computations
        self.committor  = committor
        
        ## We also run the first forward-backward pass here, using the torch_config 
        ## saved in our sampler
        #Zero out any gradients
        self.committor.zero_grad()

        #No more backprop because gradients will never be needed during sampling
        #Forward pass
        self.out = self.committor(self.sampler.torch_config)
        
        #Backprop to compute gradients w.r.t. x
        self.out.backward()

    def runSimulation(self):
        ## Create storage entries
        
        #Configurations for one mini-batch
        configs = torch.zeros(self.batch_size,self.dimN)
        
        #Gradient of committor w.r.t. x for every x in configs
        grads = torch.zeros(self.batch_size,self.dimN)
        
        
        for i in range(self.batch_size):
            for j in range(self.period):
                #Take one step
                self.sampler.step()
                
                #Save config
                self.sampler.save()        
                
            #No more backprop because gradients will never be needed during sampling
            #Zero out any gradients
            self.committor.zero_grad()
            
            #Forward pass
            self.out = self.committor(self.sampler.torch_config)
            
            #Backprop to compute gradients of x
            self.out.backward()
            
            if torch.sum(torch.isnan(self.out)) > 0:
                raise ValueError("Committor value is NaN!")
            else:
                #Compute all for all storage entries
                configs[i,:] = self.sampler.torch_config
                grads[i,:] = torch.autograd.grad(self.committor(self.sampler.torch_config), self.sampler.torch_config, create_graph=True)[0]
        
        #Zero out any gradients in the parameters as the last remaining step
        self.committor.zero_grad()

        return configs, grads

class EXPReweightSimulation:
    def __init__(self, sampler, committor, period, batch_size, dimN):
        ## Store the MD/MC Simulator, which samples our data
        self.sampler = sampler
        ## The number of timesteps we do per iteration of the optimization
        self.period = period
        ## The number of iterations we do per optimization step
        self.batch_size = batch_size
        ## The size of the problem
        self.dimN = dimN

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
        ## Create storage entries
        
        #Configurations for one mini-batch
        configs = torch.zeros(self.batch_size,self.dimN)
        
        #Gradient of committor w.r.t. x for every x in configs
        grads = torch.zeros(self.batch_size,self.dimN)
        
        #1/c(x) constant for every x in configs
        reciprocal_normconstant = torch.zeros(self.batch_size)
        
        #w_{l+1}/w_{l} computed for every x in configs
        fwd_weightfactor = torch.zeros(self.batch_size, 1)
        
        #w_{l-1}/w_{l} computed for every x in configs
        bwrd_weightfactor = torch.zeros(self.batch_size, 1)
        for i in range(self.batch_size):
            for j in range(self.period):
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
                
                #Compute all for all storage entries
                configs[i,:] = self.sampler.torch_config
                grads[i,:] = torch.autograd.grad(self.committor(self.sampler.torch_config), self.sampler.torch_config, create_graph=True)[0]
                reciprocal_normconstant[i] = self.sampler.reciprocal_normconstant
                fwd_weightfactor[i,:] = self.sampler.fwd_weightfactor
                bwrd_weightfactor[i,:] = self.sampler.bwrd_weightfactor
        
        #Zero out any gradients in the parameters as the last remaining step
        self.committor.zero_grad()

        return configs, grads, reciprocal_normconstant, fwd_weightfactor, bwrd_weightfactor
