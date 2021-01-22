#Import necessarry tools from torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn as nn

#Import necessarry tools from tpstorch 
from tpstorch.ml.optim import project_simplex
from tpstorch.ml import MLSamplerEXP
from tpstorch.ml.nn import CommittorLossEXP
from mullerbrown_ml import MySampler
import numpy as np

#Import any other thing
import tqdm, sys

class CommittorNet(nn.Module):
    def __init__(self, d, num_nodes, unit=torch.relu):
        super(CommittorNet, self).__init__()
        self.num_nodes = num_nodes
        self.d = d
        self.unit = unit
        self.lin1 = nn.Linear(d, num_nodes, bias=True)
        #self.lin12 = nn.Linear(num_nodes, num_nodes, bias=True)
        self.lin2 = nn.Linear(num_nodes, 1, bias=False)
        self.thresh = torch.sigmoid
        self.project()
        self.broadcast()

    def forward(self, x):
        #At the moment, x is flat. So if you want it to be 2x1 or 3x4 arrays, then you do it here!
        x = self.lin1(x)
        x = self.unit(x)
        # x = self.lin12(x)
        # x = self.unit(x)
        x = self.lin2(x)
        x = self.thresh(x)
        return x
    
    def broadcast(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                dist.broadcast(param.data,src=0)
    def project(self):
        #Project the coefficients so that they are make the output go from zero to one
        with torch.no_grad():
            self.lin2.weight.data = project_simplex(self.lin2.weight.data)

# This is a sketch of what it should be like, but basically have to also make committor play nicely
# within function
# have omnious committor part that actual committor overrides?
class MullerBrown(MySampler):
    def __init__(self,param,config,rank,dump,beta,kappa,mpi_group,committor,save_config=False):
        super(MullerBrown, self).__init__(param,config.detach().clone(),rank,dump,beta,kappa,mpi_group)

        self.committor = committor

        self.save_config = save_config
        self.timestep = 0

        #Save config size and its flattened version
        self.config_size = config.size()
        self.flattened_size = config.flatten().size()
        self.torch_config = config

    def initialize_from_torchconfig(self, config):
        # Don't have to worry about all that much all, can just set it
        self.setConfig(config)

    def step(self, committor_val, onlytst=False):
        with torch.no_grad():
            config_test = torch.zeros_like(self.torch_config)
            self.propose(config_test, committor_val, onlytst)
            committor_val_ = self.committor(config_test)
            self.acceptReject(config_test, committor_val_, onlytst, True)
        self.torch_config = (self.getConfig().flatten()).detach().clone()
        self.torch_config.requires_grad_()
        try:
            self.torch.grad.data.zero_()
        except:
            pass

    def step_unbiased(self):
        with torch.no_grad():
            config_test = torch.zeros_like(self.torch_config)
            self.propose(config_test, 0.0, False)
            self.acceptReject(config_test, 0.0, False, False)
        self.torch_config = (self.getConfig().flatten()).detach().clone()
        self.torch_config.requires_grad_()
        try:
            self.torch.grad.data.zero_()
        except:
            pass

    def save(self):
        self.timestep += 1
        if self.save_config:
            if self.timestep % 100 == 0:
                self.dumpConfig(0)


class MullerBrownLoss(CommittorLossEXP):
    def __init__(self, lagrange_bc, batch_size,start,end,radii,world_size,n_boundary_samples,react_configs,prod_configs,committor_start,committor_rate,final_count, k_committor, sim_committor, committor_trials, mode="random", ref_index=None, batch_size_bc = 0.5, batch_size_cm = 0.5):
        super(MullerBrownLoss,self).__init__()
        # Committor loss stuff
        self.cm_loss = torch.zeros(1)
        self.committor_start = committor_start
        self.final_count = final_count
        self.committor_rate = committor_rate
        self.committor_configs = torch.zeros(int((self.final_count-self.committor_start)/committor_rate+2),react_configs.shape[1], dtype=torch.float) 
        self.committor_configs_values = torch.zeros(int((self.final_count-self.committor_start)/committor_rate+2), dtype=torch.float)
        self.committor_configs_count = 0
        self.k_committor = k_committor
        self.sim_committor = sim_committor
        self.committor_trials = committor_trials
        # Batch size for BC, CM losses
        self.batch_size_bc = batch_size_bc
        self.batch_size_cm = batch_size_cm
        # Other stuff
        self.lagrange_bc = lagrange_bc
        self.start = start
        self.end = end
        self.radii = radii
        self.world_size = world_size
        self.react_configs = react_configs
        self.prod_configs = prod_configs
        self.react_lambda = torch.zeros(1)
        self.prod_lambda = torch.zeros(1)
        self.mean_recipnormconst = torch.zeros(1)
        #Storing a 1D Tensor of reweighting factors
        self.reweight = [torch.zeros(1) for i in range(dist.get_world_size())]
        #Choose whether to sample window references randomly or not
        self.mode = mode
        if mode != "random":
            if ref_index is None or ref_index < 0:
                raise ValueError("For non-random choice of window reference, you need to set ref_index!")
            else:
                self.ref_index = torch.tensor(ref_index)

    def compute_bc(self, committor, configs, invnormconstants):
        #Assume that first dimension is batch dimension
        loss_bc = torch.zeros(1)
        # if dist.get_rank() == 0:
            # print(self.react_configs)
            # print(committor(self.react_configs))
            # print(torch.mean(committor(self.react_configs)))
            # print(self.prod_configs)
            # print(committor(self.prod_configs))
            # print(torch.mean(committor(self.prod_configs)))
        self.react_lambda = self.react_lambda.detach()
        self.prod_lambda = self.prod_lambda.detach()
        indices_react = torch.randperm(len(self.react_configs))[:int(0.5*len(self.react_configs))]
        indices_prod = torch.randperm(len(self.prod_configs))[:int(0.5*len(self.prod_configs))]
        react_penalty = torch.mean(committor(self.react_configs[indices_react,:]))
        prod_penalty = torch.mean(1.0-committor(self.prod_configs[indices_prod,:]))
        #self.react_lambda -= self.lagrange_bc*react_penalty
        #self.prod_lambda -= self.lagrange_bc*prod_penalty
        loss_bc += 0.5*self.lagrange_bc*react_penalty**2-self.react_lambda*react_penalty
        loss_bc += 0.5*self.lagrange_bc*prod_penalty**2-self.prod_lambda*prod_penalty
        return loss_bc/self.world_size

    def computeZl(self,k,fwd_meanwgtfactor,bwrd_meanwgtfactor):
        with torch.no_grad():
            empty = []
            for l in range(dist.get_world_size()):
                if l > k:
                    empty.append(torch.prod(fwd_meanwgtfactor[k:l]))
                elif l < k:
                    empty.append(torch.prod(bwrd_meanwgtfactor[l:k]))
                else:
                    empty.append(torch.tensor(1.0))
            return torch.tensor(empty).flatten()

    def forward(self, gradients, committor, config, invnormconstants, fwd_weightfactors, bwrd_weightfactors, reciprocal_normconstants, counter, config_current):
        self.main_loss = self.compute_loss(gradients, invnormconstants)
        self.bc_loss = self.compute_bc(committor, config, invnormconstants)
        self.cm_loss = self.compute_cl(config_current, committor, counter)
        #renormalize losses
        #get prefactors
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
        self.reweight = self.computeZl(self.ref_index,fwd_meanwgtfactor,bwrd_meanwgtfactor)
        self.reweight.div_(torch.sum(self.reweight))  #normalize
        
        #Use it first to compute the mean inverse normalizing constant
        mean_recipnormconst = torch.mean(reciprocal_normconstants)#invnormconstants)
        mean_recipnormconst.mul_(self.reweight[dist.get_rank()])

        #All reduce the mean invnormalizing constant
        dist.all_reduce(mean_recipnormconst)

        #renormalize main_loss
        self.main_loss *= self.reweight[dist.get_rank()]
        dist.all_reduce(self.main_loss)
        self.main_loss /= mean_recipnormconst

        #normalize bc_loss
        dist.all_reduce(self.bc_loss)

        #normalize cm_loss
        dist.all_reduce(self.cm_loss)
        return self.main_loss+self.bc_loss+self.cm_loss

    def myprod_checker(self, config):
        end = torch.tensor([[0.5,0.0]])
        radii = 0.025
        end_ = config-end
        end_ = end_.pow(2).sum()**0.5
        if end_ <= radii:
            return True
        else:
            return False

    def myreact_checker(self, config):
        start = torch.tensor([[-0.5,1.5]])
        radii = 0.025
        start_ = config-start
        start_ = start_.pow(2).sum()**0.5
        if start_ <= radii:
            return True
        else:
            return False

    def compute_cl(self, config, committor, counter):
        loss_cl = torch.zeros(1)
        if(counter<self.committor_start):
            return loss_cl
        elif(counter==self.committor_start):
            # Generate first committor config
            counts = []
            for i in range(self.committor_trials):
                self.sim_committor.initialize_from_torchconfig(config.detach().clone())
                hitting = False

                #Run simulation and stop until it falls into the product or reactant state
                while hitting is False:
                    self.sim_committor.step_unbiased()
                    if self.myreact_checker(self.sim_committor.getConfig()):
                        hitting = True
                        counts.append(0)
                    elif self.myprod_checker(self.sim_committor.getConfig()):
                        hitting = True
                        counts.append(1)

            counts = np.array(counts)
            self.committor_configs_values[0] = np.mean(counts) 
            self.committor_configs[0] = config.detach().clone()
            self.committor_configs_count += 1
            # Now compute loss
            committor_penalty = committor(self.committor_configs[0])-self.committor_configs_values[0]
            loss_cl += 0.5*self.k_committor*committor_penalty**2
            return loss_cl/self.world_size
        else:
            # Generate new committor configs and keep on generating the loss
            if(counter%self.committor_rate==0):
                # Generate first committor config
                counts = []
                for i in range(self.committor_trials):
                    self.sim_committor.initialize_from_torchconfig(config.detach().clone())
                    hitting = False

                    #Run simulation and stop until it falls into the product or reactant state
                    while hitting is False:
                        self.sim_committor.step_unbiased()
                        if self.myreact_checker(self.sim_committor.getConfig()):
                            hitting = True
                            counts.append(0)
                        elif self.myprod_checker(self.sim_committor.getConfig()):
                            hitting = True
                            counts.append(1)

                counts = np.array(counts)
                configs_count = self.committor_configs_count
                self.committor_configs_values[configs_count] = np.mean(counts) 
                self.committor_configs[configs_count] = config.detach().clone()
                self.committor_configs_count += 1
            # Compute loss
            indices_committor = torch.randperm(self.committor_configs_count)[:int(0.5*self.committor_configs_count)]
            if self.committor_configs_count == 0:
                indices_committor = 0
            committor_penalty = torch.mean(committor(self.committor_configs[indices_committor])-self.committor_configs_values[indices_committor])
            print(str(dist.get_rank())+" "+str(committor_penalty))
            loss_cl += 0.5*self.k_committor*committor_penalty**2
            return loss_cl/self.world_size
