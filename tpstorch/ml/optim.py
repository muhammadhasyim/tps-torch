import torch
from torch.optim import Optimizer, SGD
from torch.optim.optimizer import required
import torch.distributed as dist
from pymbar import timeseries

#A helper function to compute the un-normalized reweighting factors
def computeFlk(k,fwd_meanwgtfactor,bwrd_meanwgtfactor):
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

class UnweightedSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum). 
    
    This implementation has an additional step which is to collect gradients computed in different 
    MPI processes and just average them. This is useful when you're running many unbiased simulations  
    
    Any more detailed implementation should be consulted on torch.optim.SGD
    """

    def __init__(self, params, sampler=required, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
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
        super(UnweightedSGD, self).__init__(params, defaults)
       
    def __setstate__(self, state):
        super(UnweightedSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

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

                #This is the new part from the original SGD implementation
                #We just do an all reduce on the gradients
                dist.all_reduce(d_p)
                d_p.div_(dist.get_world_size())
                
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

        return loss



class EXPReweightSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    
    This implementation has an additional step which is reweighting the computed 
    gradients through free-energy estimation techniques. Currently only implementng
    exponential averaging (EXP) because it is cheap. 
    
    """

    def __init__(self, params, sampler=required, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
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
        random_index = torch.randint(low=0,high=dist.get_world_size(),size=(1,))
        dist.broadcast(random_index, src=0)
        
        #Computing the reweighting factors, z_l in  our notation
        self.reweight = computeFlk(random_index.item(),fwd_meanwgtfactor,bwrd_meanwgtfactor)#newcontainer)
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

                #What we need to do now is to compute with its respective weight
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
