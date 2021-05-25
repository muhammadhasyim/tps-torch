#holds the loss function. And if we get the time, the custom neural net layer with the string path
import torch
import torch.nn as nn

import numpy as np

from torch.nn.modules.loss import _Loss
#import scipy.sparse.linalg

from tpstorch import _rank, _world_size
from tpstorch import dist


from numpy.linalg import svd
import scipy.linalg

#Helper function to obtain null space
def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    ss = s[nnz:]
    return ns, ss

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

class FTSCommittorLoss(_Loss):
    r"""Loss function which implements the MSE loss for the committor function. 
        
        This loss function automatically collects the approximate values of the committor around a string.

        Args:
            fts_sampler (tpstorch.MLSampler): the MC/MD sampler to perform biased simulations in the string.
            
            lambda_fts (float): the penalty strength of the MSE loss. Defaults to one. 
            
            fts_start (int): iteration number where we start collecting samples for the committor loss. 

            fts_end (int): iteration number where we stop collecting samples for the committor loss

            fts_rate (int): sampling rate for collecting committor samples during the iterations. 

            fts_max_steps (int): number of maximum timesteps to compute the committor per initial configuration.
            
            fts_min_count (int): minimum number of rejection counts before we stop the simulation. 

            batch_size_fts (float): size of mini-batch used during training, expressed as the fraction of total batch collected at that point. 
    """
    def __init__(self, fts_sampler, committor, dimN, lambda_fts=1.0, fts_start=200, fts_end=2000000, fts_rate=100, fts_max_steps=10**6, fts_min_count=10, batch_size_fts=0.5):
        super(FTSCommittorLoss, self).__init__()
        
        self.fts_loss = torch.zeros(1)
        self.committor = committor

        self.fts_sampler = fts_sampler
        self.fts_start = fts_start
        self.fts_end = fts_end
        self.lambda_fts = lambda_fts
        self.fts_rate = fts_rate
        self.batch_size_fts = batch_size_fts
        
        if fts_sampler.torch_config.shape == torch.Size([1]):
            self.fts_configs = torch.zeros(int((self.fts_end-self.fts_start)/fts_rate+2), dimN, dtype=torch.float) 
        else:
            self.fts_configs = torch.zeros(int((self.fts_end-self.fts_start)/fts_rate+2), dimN, dtype=torch.float) 
        self.fts_configs_values = torch.zeros(int((self.fts_end-self.fts_start)/fts_rate+2), dtype=torch.float)
        self.fts_configs_count = 0
    
        self.min_count = fts_min_count
        self.max_steps = fts_max_steps

    def runSimulation(self, strings):
        with torch.no_grad():
            #Initialize
            self.fts_sampler.committor_list[0] = 0.0
            self.fts_sampler.committor_list[-1] = 1.0
            for i in range(_world_size):
                self.fts_sampler.rejection_count[i] = 0
                if i > 0:
                    self.fts_sampler.committor_list[i]= self.committor(0.5*(strings[i-1]+strings[i])).item()
            inftscell = self.fts_sampler.checkFTSCell(self.committor(self.fts_sampler.getConfig().flatten()), _rank, _world_size)
            if inftscell:
                pass
            else:
                self.fts_sampler.setConfig(strings[_rank])
            
            for i in range(self.max_steps):
                self.fts_sampler.step()
                #Here we check if we have enough rejection counts, 
                #If we don't, then we need to run the simulation a little longer
                if _rank == 0 and self.fts_sampler.rejection_count[_rank+1].item() >= self.min_count:
                        break
                elif _rank == _world_size-1 and self.fts_sampler.rejection_count[_rank-1].item() >= self.min_count:
                        break
                elif self.fts_sampler.rejection_count[_rank-1].item() >= self.min_count and self.fts_sampler.rejection_count[_rank+1].item() >= self.min_count:
                        break
            #Since simulations may run in un-equal amount of times, we have to normalize rejection counts by the number of timesteps taken
            self.fts_sampler.normalizeRejectionCounts()
    
    @torch.no_grad()
    def compute_qalpha(self):
    #def computeZl(self,rejection_counts):
        """ Computes the reweighting factor z_l by solving an eigenvalue problem
            
            Args:
                rejection_counts (torch.Tensor): an array of rejection counts, i.e., how many times
                a system steps out of its cell in the FTS smulation, stored by an MPI process. 
            
            Formula and maths is based on our paper. 
        """
        qalpha = torch.linspace(0,1,_world_size)
        #Set the row according to rank
        Kmatrix = torch.zeros(_world_size,_world_size)
        Kmatrix[_rank] = self.fts_sampler.rejection_count
        
        #All reduce to collect the results
        dist.all_reduce(Kmatrix)
        
        #Finallly, build the final matrix
        Kmatrix = Kmatrix.t()-torch.diag(torch.sum(Kmatrix,dim=1))
        
        #Compute the reweighting factors using an eigensolver
        #w, v = scipy.sparse.linalg.eigs(A=Kmatrix.numpy(),k=1, which='SM')
        v, w = nullspace(Kmatrix.numpy(), atol=1e-6, rtol=0)
        index = np.argmin(w)
        zl = np.real(v[:,index])
        
        #Normalize
        zl = zl/np.sum(zl)  
        
        #Alright, now compute approximate committor values around the string
        #Build a new matrix
        eGalpha = torch.zeros(_world_size-2,_world_size-2)
        if _rank > 0 and _rank < _world_size-1:
            i = _rank-1
            eGalpha[i,i] = float(-zl[_rank-1]-zl[_rank+1])
            if i != 0:
                eGalpha[i,i-1] = float(zl[_rank-1])
            if i != _world_size-3:
                eGalpha[i,i+1] = float(zl[_rank+1])
        dist.all_reduce(eGalpha)
        b = np.zeros(_world_size-2)
        b[-1] = -zl[-1]
        qalpha[1:-1] = torch.from_numpy(scipy.linalg.solve(a=eGalpha.numpy(),b=b)) 
        return qalpha[_rank]
    
    def compute_fts(self,counter, strings):#,initial_config):
        """Computes the committor loss function 
            TO DO: Complete this docstrings 
        """
        #Initialize loss to zero
        loss_fts = torch.zeros(1)
        
        if ( counter < self.fts_start):
            #Not the time yet to compute the committor loss
            return loss_fts
        elif (counter==self.fts_start):
            
            #Generate the first committor sample
            self.runSimulation(strings)
            print("Rank [{}] finishes simulation for committor calculation".format(_rank))
            
            #Save the committor values and initial configuration 
            self.fts_configs_values[0] = self.compute_qalpha()
            self.fts_configs[0] = strings[_rank].detach().clone()
            self.fts_configs_count += 1
            
            # Now compute loss
            committor_penalty = torch.mean((self.committor(self.fts_configs[0])-self.fts_configs_values[0])**2)
            loss_fts += 0.5*self.lambda_fts*committor_penalty
            
            #Collect all the results
            dist.all_reduce(loss_fts)
            
            return loss_fts/_world_size
        else:
            if counter % self.fts_rate==0 and counter < self.fts_end:
                # Generate new committor configs and keep on generating the loss
                self.runSimulation(strings)
                print("Rank [{}] finishes simulation for committor calculation".format(_rank))
                configs_count = self.fts_configs_count
                self.fts_configs_values[configs_count] = self.compute_qalpha()
                self.fts_configs[configs_count] = strings[_rank].detach().clone()
                self.fts_configs_count += 1
            
            # Compute loss by sub-sampling however many batches we have at the moment
            indices_committor = torch.randperm(self.fts_configs_count)[:int(self.batch_size_fts*self.fts_configs_count)]
            if self.fts_configs_count == 1:
                indices_committor = 0
            committor_penalty = torch.mean((self.committor(self.fts_configs[indices_committor])-self.fts_configs_values[indices_committor])**2)
            loss_fts += 0.5*self.lambda_fts*committor_penalty
            
            #Collect all the results
            dist.all_reduce(loss_fts)
            return loss_fts/_world_size
    
    def forward(self, counter, strings):
        self.fts_loss = self.compute_fts(counter, strings)
        return self.fts_loss


class CommittorLoss(_Loss):
    r"""Loss function which implements the MSE loss for the committor function. 
        
        This loss function automatically collects the committor values through brute-force simulation.

        Args:
            cl_sampler (tpstorch.MLSampler): the MC/MD sampler to perform unbiased simulations.
            
            committor (tpstorch.nn.Module): the committor function, represented as a neural network. 
            
            lambda_cl (float): the penalty strength of the MSE loss. Defaults to one. 
            
            cl_start (int): iteration number where we start collecting samples for the committor loss. 

            cl_end (int): iteration number where we stop collecting samples for the committor loss

            cl_rate (int): sampling rate for collecting committor samples during the iterations. 

            cl_trials (int): number of trials to compute the committor per initial configuration.

            batch_size_cl (float): size of mini-batch used during training, expressed as the fraction of total batch collected at that point. 
    """
    def __init__(self, cl_sampler, committor, lambda_cl=1.0, cl_start=200, cl_end=2000000, cl_rate=100, cl_trials=50, batch_size_cl=0.5):
        super(CommittorLoss, self).__init__()
        
        self.cl_loss = torch.zeros(1)
        self.committor = committor 

        self.cl_sampler = cl_sampler
        self.cl_start = cl_start
        self.cl_end = cl_end
        self.lambda_cl = lambda_cl
        self.cl_rate = cl_rate
        self.cl_trials = cl_trials
        self.batch_size_cl = batch_size_cl
        
        if cl_sampler.torch_config.shape == torch.Size([1]):
            self.cl_configs = torch.zeros(int((self.cl_end-self.cl_start)/cl_rate+2), cl_sampler.torch_config.shape[0], dtype=torch.float) 
        else:
            self.cl_configs = torch.zeros(int((self.cl_end-self.cl_start)/cl_rate+2), cl_sampler.torch_config.shape[1], dtype=torch.float) 
        self.cl_configs_values = torch.zeros(int((self.cl_end-self.cl_start)/cl_rate+2), dtype=torch.float)
        self.cl_configs_count = 0
    
    def runTrials(self, config):
        counts = []
        
        for i in range(self.cl_trials):
            self.cl_sampler.initialize_from_torchconfig(config.detach().clone())
            hitting = False
            #Run simulation and stop until it falls into the product or reactant state
            steps = 0
            while hitting is False:
                if self.cl_sampler.isReactant(self.cl_sampler.getConfig()):
                    hitting = True
                    counts.append(0)
                elif self.cl_sampler.isProduct(self.cl_sampler.getConfig()):
                    hitting = True
                    counts.append(1)
                self.cl_sampler.step_unbiased()
                steps += 1
        return np.array(counts)
    
    def compute_cl(self,counter,initial_config):
        """Computes the committor loss function 
            TO DO: Complete this docstrings 
        """
        #Initialize loss to zero
        loss_cl = torch.zeros(1)
        
        if ( counter < self.cl_start):
            #Not the time yet to compute the committor loss
            return loss_cl
        elif (counter==self.cl_start):
            
            #Generate the first committor sample
            counts = self.runTrials(initial_config)
            print("Rank [{}] finishes committor calculation: {} +/- {}".format(_rank, np.mean(counts), np.std(counts)/len(counts)**0.5))
            
            #Save the committor values and initial configuration 
            self.cl_configs_values[0] = np.mean(counts) 
            self.cl_configs[0] = initial_config.detach().clone()
            self.cl_configs_count += 1
            
            # Now compute loss
            committor_penalty = torch.mean((self.committor(self.cl_configs[0])-self.cl_configs_values[0])**2)
            loss_cl += 0.5*self.lambda_cl*committor_penalty
            #Collect all the results
            dist.all_reduce(loss_cl)
            return loss_cl/_world_size
        else:
            if counter % self.cl_rate==0 and counter < self.cl_end:
                # Generate new committor configs and keep on generating the loss
                counts = self.runTrials(initial_config)
                print("Rank [{}] finishes committor calculation: {} +/- {}".format(_rank, np.mean(counts), np.std(counts)/len(counts)**0.5))
                configs_count = self.cl_configs_count
                self.cl_configs_values[configs_count] = np.mean(counts) 
                self.cl_configs[configs_count] = initial_config.detach().clone()
                self.cl_configs_count += 1
            
            # Compute loss by sub-sampling however many batches we have at the moment
            indices_committor = torch.randperm(self.cl_configs_count)[:int(self.batch_size_cl*self.cl_configs_count)]
            if self.cl_configs_count == 1:
                indices_committor = 0
            committor_penalty = torch.mean((self.committor(self.cl_configs[indices_committor])-self.cl_configs_values[indices_committor])**2)
            loss_cl += 0.5*self.lambda_cl*committor_penalty
            
            #Collect all the results
            dist.all_reduce(loss_cl)
            return loss_cl/_world_size
    
    def forward(self, counter, initial_config):
        self.cl_loss = self.compute_cl(counter, initial_config)
        return self.cl_loss


class _BKELoss(_Loss):
    r"""Base classs for computing the loss function corresponding to the variational form 
        of the Backward Kolmogorov Equation. This base class includes default implementation 
        for boundary conditions. 

    Args:
        bc_sampler (tpstorch.MLSamplerEXP): the MD/MC sampler used for obtaining configurations in 
            product and reactant basin.  
        
        committor (tpstorch.nn.Module): the committor function, represented as a neural network. 
        
        lambda_A (float): penalty strength for enforcing boundary conditions at the reactant basin. 
        
        lambda_B (float): penalty strength for enforcing boundary conditions at the product basin. 
            If None is given, lambda_B=lambda_A
        
        start_react (torch.Tensor): starting configuration to sample reactant basin. 
        
        start_prod (torch.Tensor): starting configuration to sample product basin.
       
        n_bc_samples (int, optional): total number of samples to collect at both product and 
            reactant basin. 

        bc_period (int, optional): the number of timesteps to collect one configuration during 
            sampling at either product and reactant basin.

        batch_size_bc (float, optional): size of mini-batch for the boundary condition loss during 
            gradient descent, expressed as fraction of n_bc_samples.
    """

    def __init__(self, bc_sampler, committor, lambda_A, lambda_B, start_react, 
                 start_prod, n_bc_samples=320, bc_period=10, batch_size_bc=0.1):
        super(_BKELoss, self).__init__()
        
        self.main_loss = torch.zeros(1)
        self.bc_loss = torch.zeros(1)
        
        self.committor = committor
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.n_bc_samples = n_bc_samples
        self.batch_size_bc = batch_size_bc
        
        self.prod_configs = torch.zeros(self.n_bc_samples, start_prod.shape[0]*start_prod.shape[1], dtype=torch.float)
        self.react_configs = torch.zeros_like(self.prod_configs)
        self.zl = [torch.zeros(1) for i in range(_world_size)]
            
        #Sample product basin first
        if _rank == 0:
            print("Sampling the product basin",flush=True)
        bc_sampler.setConfig(start_prod)
        for i in range(self.n_bc_samples):
            for j in range(bc_period):
                bc_sampler.step_bc()
            self.prod_configs[i] = bc_sampler.getConfig()
        
        #Next, sample reactant basin
        if _rank == 0:
            print("Sampling the reactant basin",flush=True)
        bc_sampler.setConfig(start_react)
        for i in range(self.n_bc_samples):
            for j in range(bc_period):
                bc_sampler.step_bc()
            self.react_configs[i] = bc_sampler.getConfig()

    def compute_bc(self):
        """Computes the loss due to the boundary conditions.
            
            Default implementation is to apply the penalty method, where the loss at every MPI 
            process is computed as: 

            .. math::
                \ell_{BC,i} = \frac{\lambda_A}{2} \avg_{x \in M_A} (q(x))^2 
                                + \frac{\lambda_B}{2} \avg_{x \in M_B} (1-q(x))^2
            
            where 'i' is the index of the MPI process, :math: 'q(x)' is the 
            committor function, :math: 'M_A' is the mini-batch for the product 
            basin and 'M_B' is the the mini-batch for the reactant basin. 
            
            The result is then collected (MPI collective communication) as follows:

            .. math::
                \ell_{BC} = {1}{S} \sum_{i=0}^{S-1} \ell_{BC,i}
            
            where 'S' is the MPI world size. 

            Note that PyTorch does not track arithmetic operations during MPI
            collective calls. Thus, the last sum is not reflected in the
            computational graph tracked by individual MPI process. The final 
            gradients will be collected in each respective optimizer.
            
            #To do: create a mode where configurations in either basin is always
            #always sampled on-the-fly.
        """
        
        loss_bc = torch.zeros(1)
        
        #Compute random indices to sub-sample the list of reactant and product
        #configurations
        indices_react = torch.randperm(len(self.react_configs))[:int(self.batch_size_bc*len(self.react_configs))]
        indices_prod = torch.randperm(len(self.prod_configs))[:int(self.batch_size_bc*len(self.prod_configs))]
        
        react_penalty = torch.mean(self.committor(self.react_configs[indices_react,:])**2)
        
        prod_penalty = torch.mean((1.0-self.committor(self.prod_configs[indices_prod,:]))**2)
        
        loss_bc += 0.5*self.lambda_A*react_penalty
        
        loss_bc += 0.5*self.lambda_B*prod_penalty
        return loss_bc/_world_size
    
    @torch.no_grad()
    def computeZl(self):
        raise NotImplementedError
    
    def compute_bkeloss(self):
        raise NotImplementedError

    def forward(self):
        r"""Default implementation computes the boundary condition loss only"""
        self.bc_loss = self.compute_bc()
        return self.bc_loss

class BKELossEXP(_BKELoss):
    r"""Loss function corresponding to the variational form of the Backward Kolmogorov Equation, 
        which includes reweighting by exponential (EXP) averaging. 

    Args:
        bc_sampler (tpstorch.MLSamplerEXP): the MD/MC sampler used for obtaining configurations in 
            product and reactant basin.  
        
        committor (tpstorch.nn.Module): the committor function, represented as a neural network. 
        
        lambda_A (float): penalty strength for enforcing boundary conditions at the reactant basin. 
        
        lambda_B (float): penalty strength for enforcing boundary conditions at the product basin. 
            If None is given, lambda_B=lambda_A
        
        start_react (torch.Tensor): starting configuration to sample reactant basin. 
        
        start_prod (torch.Tensor): starting configuration to sample product basin.
       
        n_bc_samples (int, optional): total number of samples to collect at both product and 
            reactant basin. 

        bc_period (int, optional): the number of timesteps to collect one configuration during 
            sampling at either product and reactant basin.

        batch_size_bc (float, optional): size of mini-batch for the boundary condition loss during 
            gradient descent, expressed as fraction of n_bc_samples.

        mode (string, optional): the mode for EXP reweighting. If mode is 'random', then the 
            reference umbrella window is chosen randomly at every iteration. If it's not random, 
            then ref_index must be supplied.

        ref_index (int, optional): a fixed chosen umbrella window for computing the reweighting 
            factors. 
    """

    def __init__(self, bc_sampler, committor, lambda_A, lambda_B, start_react, 
                 start_prod, n_bc_samples=320, bc_period=10, batch_size_bc=0.1, 
                 mode='random', ref_index=None):
        super(BKELossEXP, self).__init__(bc_sampler, committor, lambda_A, lambda_B, start_react, 
                                        start_prod, n_bc_samples, bc_period, batch_size_bc)
        self.mode = mode 
        if self.mode != 'random':
            if ref_index is not None:
                self.ref_index = int(ref_index)
            else:
                raise TypeError
    
    @torch.no_grad()
    def computeZl(self, fwd_weightfactors, bwrd_weightfactors):
        """Computes the reweighting factor z_L needed for computing gradient
            averages.
            
            Args:
                fwd_weightfactors (torch.Tensor): mini-batch of w_{l+1}/w_{l}, which are forward 
                    ratios of the umbrella potential Boltzmann factors stored by the l-th umbrella 
                    window/MPI process
                
                bwrd_weightfactors (torch.Tensor): mini-batch of w_{l-1}/w_{l}, which are forward 
                    ratios of the umbrella potential Boltzmann factors stored by the l-th umbrella 
                    window/MPI process
            
            Formula is based on Eq. REF of our paper. 
        """
        
        #Randomly select a window as a free energy reference and broadcast that index across all processes
        if self.mode == "random":
            self.ref_index = torch.randint(low=0,high=_world_size,size=(1,))
            dist.broadcast(self.ref_index, src=0)
       
        #Compute the average of forward and backward ratios of Boltzmann factors
        fwd_meanwgtfactor = [torch.zeros(1) for i in range(_world_size)]
        dist.all_gather(fwd_meanwgtfactor,torch.mean(fwd_weightfactors))
        fwd_meanwgtfactor = torch.tensor(fwd_meanwgtfactor[:-1])

        bwrd_meanwgtfactor = [torch.zeros(1) for i in range(_world_size)] 
        dist.all_gather(bwrd_meanwgtfactor,torch.mean(bwrd_weightfactors))
        bwrd_meanwgtfactor = torch.tensor(bwrd_meanwgtfactor[1:])
        
        #Compute the reweighting factor
        zl = []
        for l in range(_world_size):
            if l > self.ref_index:
                zl.append(torch.prod(fwd_meanwgtfactor[self.ref_index:l]))
            elif l < self.ref_index:
                zl.append(torch.prod(bwrd_meanwgtfactor[l:self.ref_index]))
            else:
                zl.append(torch.tensor(1.0))
        
        #Normalize the reweighting factor
        zl = torch.tensor(zl).flatten()
        zl.div_(torch.sum(zl))
        
        return zl
    
    def compute_bkeloss(self, gradients, inv_normconstants, fwd_weightfactors, bwrd_weightfactors):
        """Computes the loss corresponding to the varitional form of the BKE including 
            the EXP reweighting factors. 
            
            Independent computation is first done on individual MPI process. First, we compute 
            the following quantities at every 'l'-th MPI process: 

            .. math::
                L_l = \frac{1}{2} \sum_{x \in M_l} |\grad q(x)|^2/c(x) ,
                
                c_l = \sum_{ x \in M_l} 1/c(x) ,
            
            where :math: $M_l$ is the mini-batch collected by the l-th MPI
            process. We then collect the computation to compute the main loss as

            .. math::
                \ell_{main} = \frac{\sum_{l=1}^{S-1} L_l z_l)}{\sum_{l=1}^{S-1} c_l z_l)}
            where :math: 'S' is the MPI world size. 

            Args:
                gradients (torch.Tensor): mini-batch of \grad q(x). First dimension is the size of
                    the mini-batch while the second is system size (flattened).
                
                inv_normconstants (torch.Tensor): mini-batch of 1/c(x).

                fwd_weightfactors (torch.Tensor): mini-batch of w_{l+1}/w_{l}, which are forward 
                    ratios of the umbrella potential Boltzmann factors stored by the l-th umbrella 
                    window/MPI process
                
                bwrd_weightfactors (torch.Tensor): mini-batch of w_{l-1}/w_{l}, which are forward 
                    ratios of the umbrella potential Boltzmann factors stored by the l-th umbrella 
                    window/MPI process
            
            Note that PyTorch does not track arithmetic operations during MPI
            collective calls. Thus, the last sum containing L_l is not reflected 
            in the computational graph tracked by individual MPI process. The 
            final gradients will be collected in each respective optimizer.
        """
        main_loss = torch.zeros(1) 
        
        #Compute the first part of the loss
        main_loss =  0.5*torch.sum(torch.mean(gradients*gradients*inv_normconstants.view(-1,1),dim=0));
        
        #Computing the reweighting factors, z_l in  our notation
        self.zl = self.computeZl(fwd_weightfactors, bwrd_weightfactors)
        
        #Use it first to compute the mean inverse normalizing constant 
        mean_recipnormconst = torch.mean(inv_normconstants)
        mean_recipnormconst.mul_(self.zl[_rank])
        #All reduce the mean invnormalizing constant
        dist.all_reduce(mean_recipnormconst)
        
        #renormalize main_loss
        main_loss *= self.zl[_rank]
        dist.all_reduce(main_loss)
        main_loss /= mean_recipnormconst
        return main_loss

    def forward(self, gradients, inv_normconstants, fwd_weightfactors, bwrd_weightfactors):
        self.main_loss = self.compute_bkeloss(gradients, inv_normconstants, fwd_weightfactors, bwrd_weightfactors)
        self.bc_loss = self.compute_bc()
        return self.main_loss+self.bc_loss

class BKELossFTS(_BKELoss):
    r"""Loss function corresponding to the variational form of the Backward 
        Kolmogorov Equation, which includes reweighting by exponential (EXP)
        averaging. 

    Args:
        bc_sampler (tpstorch.MLSamplerEXP): the MD/MC sampler used for obtaining
        configurations in product and reactant basin.  
        
        committor (tpstorch.nn.Module): the committor function, represented 
        as a neural network. 
        
        lambda_A (float): penalty strength for enforcing boundary conditions at 
            the reactant basin. 
        
        lambda_B (float): penalty strength for enforcing boundary 
        conditions at the product basin. If None is given, lambda_B=lambda_A
        
        start_react (torch.Tensor): starting configuration to sample reactant 
        basin. 
        
        start_prod (torch.Tensor): starting configuration to sample product 
        basin
       
        n_bc_samples (int, optional): total number of samples to collect at both
        product and reactant basin. 

        bc_period (int, optional): the number of timesteps to collect one 
        configuration during sampling at either product and reactant basin.

        batch_size_bc (float, optional): size of mini-batch for the boundary 
        condition loss during gradient descent, expressed as fraction of 
        n_bc_samples.
    """

    def __init__(self, bc_sampler, committor, lambda_A, lambda_B, start_react, 
                 start_prod, n_bc_samples=320, bc_period=10, batch_size_bc=0.1):
        super(BKELossFTS, self).__init__(bc_sampler, committor, lambda_A, lambda_B, start_react, 
                                        start_prod, n_bc_samples, bc_period, batch_size_bc)
    
    @torch.no_grad()
    def computeZl(self,rejection_counts):
        """ Computes the reweighting factor z_l by solving an eigenvalue problem
            
            Args:
                rejection_counts (torch.Tensor): an array of rejection counts, i.e., how many times
                a system steps out of its cell in the FTS smulation, stored by an MPI process. 
            
            Formula and maths is based on our paper. 
        """
        
        #Set the row according to rank
        Kmatrix = torch.zeros(_world_size,_world_size)
        Kmatrix[_rank] = rejection_counts
        
        #All reduce to collect the results
        dist.all_reduce(Kmatrix)
        
        #Finallly, build the final matrix
        Kmatrix = Kmatrix.t()-torch.diag(torch.sum(Kmatrix,dim=1))
        
        #Compute the reweighting factors using an eigensolver
        #w, v = scipy.sparse.linalg.eigs(A=Kmatrix.numpy(),k=1, which='SM')
        v, w = nullspace(Kmatrix.numpy(), atol=1e-6, rtol=0)
        index = np.argmin(w)
        zl = np.real(v[:,index])
        
        #Normalize
        zl = zl/np.sum(zl)  
        return zl

    def compute_bkeloss(self, gradients, rejection_counts):
        """Computes the loss corresponding to the varitional form of the BKE
            including the FTS reweighting factors. 
            
            Independent computation is first done on individual MPI process. 
            First, we compute the following at every 'l'-th MPI process: 

            .. math::
                L_l = \frac{1}{2 M_l} \sum_{x \in M_l} |\grad q(x)|^2,
            
            where :math: $M_l$ is the mini-batch collected by the l-th MPI
            process. We then collect the computation to compute the main loss as

            .. math::
                \ell_{main} = \sum_{l=1}^{S-1} L_l z_l)
            
            where :math: 'S' is the MPI world size. 

            Note that PyTorch does not track arithmetic operations during MPI
            collective calls. Thus, the last sum containing L_l z_l is not reflected 
            in the computational graph tracked by individual MPI process. The 
            final gradients will be collected in each respective optimizer.
        """
        main_loss = torch.zeros(1) 
        
        #Compute the first part of the loss
        main_loss =  0.5*torch.sum(torch.mean(gradients*gradients,dim=0))
        
        #Computing the reweighting factors, z_l in  our notation
        self.zl = self.computeZl(rejection_counts)
        
        #renormalize main_loss
        main_loss *= self.zl[_rank]
        dist.all_reduce(main_loss)
        return main_loss

    def forward(self, gradients, rejection_counts):
        self.main_loss = self.compute_bkeloss(gradients, rejection_counts)
        self.bc_loss = self.compute_bc()
        return self.main_loss+self.bc_loss
