import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
#import torch.distributed as dist
import torch.optim._functional as F

from tpstorch import _rank, _world_size
from tpstorch import dist

class ParallelAdam(Optimizer):
    r"""Implements Adam algorithm.

    This implementation has an additional step which is to collect gradients computed in different 
    MPI processes and just average them. This is useful when you're running many unbiased simulations  
    
    Any more detailed implementation should be consulted on torch.optim.Adam
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(ParallelAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ParallelAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

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
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('ParallelAdam does not support sparse gradients!')
                    
                    #This is the new part from the original Adam implementation
                    #We just do an all reduce on the gradients
                    d_p = p.grad
                    dist.all_reduce(d_p)
                    
                    grads.append(d_p)
                    
                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            F.adam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   group['amsgrad'],
                   beta1,
                   beta2,
                   group['lr'],
                   group['weight_decay'],
                   group['eps']
                   )
        return loss



class ParallelSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum). 
    
    This implementation has an additional step which is to collect gradients computed in different 
    MPI processes and just average them. This is useful when you're running many unbiased simulations  
    
    Any more detailed implementation should be consulted on torch.optim.ParallelSGD
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
        super(ParallelSGD, self).__init__(params, defaults)
       
    def __setstate__(self, state):
        super(ParallelSGD, self).__setstate__(state)
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

                #This is the new part from the original ParallelSGD implementation
                #We just do an all reduce on the gradients
                dist.all_reduce(d_p)
                
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

class FTSUpdate(Optimizer):
    r"""Implements the Finite-Temperature String Method update. 
    
    It can be shown that the FTS method update is just stochastic gradient descent. Thus, one can also opt to compute the update with momentum to accelerate convergence. 
    
    """

    def __init__(self, params, sampler=required, deltatau=required, momentum=0, nesterov=False, kappa = 0.1, freeze=False):
        if deltatau is not required and deltatau < 0.0:
            raise ValueError("Invalid step size: {}".format(deltatau))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(deltatau=deltatau, momentum=momentum, nesterov=nesterov, kappa=kappa, freeze=freeze)
        super(FTSUpdate, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FTSUpdate, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, configs):
        """Performs a single optimization step.

        """

        for group in self.param_groups:
            momentum = group['momentum']
            kappa = group['kappa']
            nesterov = group['nesterov']
            freeze = group['freeze']

            for p in group['params']:
                if p.requires_grad is True:
                    print("Warning! String stored in Rank [{}] has gradient enabled. Make sure that the string is not being updated during NN training!") 

                ## (1) Compute the average configuration
                avgconfig = torch.zeros_like(p)
                avgconfig[_rank] = torch.mean(configs,dim=0)
                dist.all_reduce(avgconfig)
                
                ## (1) Stochastic Gradient Descent
                d_p = torch.zeros_like(p)
                
                d_p[1:-1] = (p[1:-1]-avgconfig[1:-1])-kappa*_world_size*(p[0:-2]-2*p[1:-1]+p[2:])
                if freeze is False:
                    d_p[0] = (p[0]-avgconfig[0])
                    d_p[-1] = (p[-1]-avgconfig[-1])
                
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['deltatau'])
                
                ## (2) Re-parameterization/Projection
                #Compute the new intermediate nodal variables
                #which doesn't obey equal arc-length parametrization
                
                alpha = torch.linspace(0,1,_world_size)
                ell_k = torch.norm(p[1:]-p[:-1],dim=1)
                ellsum = torch.sum(ell_k)
                ell_k /= ellsum
                intm_alpha = torch.zeros_like(alpha)
                for i in range(1, p.shape[0]):
                    intm_alpha[i] += ell_k[i-1]+intm_alpha[i-1]
                
                #Now interpolate back to the correct parametrization
                #TO DO: Figure out how to avoid unnecessary copy, i.e., newstring copy
                index = torch.bucketize(intm_alpha,alpha)
                newstring = torch.zeros_like(p)
                for counter, item in enumerate(index[1:-1]):
                    weight = (alpha[counter+1]-intm_alpha[item-1])/(intm_alpha[item]-intm_alpha[item-1])
                    newstring[counter+1] = torch.lerp(p[item-1],p[item],weight) 
                p[1:-1] = newstring[1:-1].detach().clone()
                del newstring
                
                #self.string[1:-1] += -self.deltatau*(self.string[1:-1]-self.avgconfig[1:-1])+self.kappa*self.deltatau*self.num_nodes*(self.string[0:-2]-2*self.string[1:-1]+self.string[2:])#-self.deltatau*self.string.grad[1:-1]*(qvals[1:-1]-self.alpha[1:-1].view(-1,1))#penalty
                #self.string[0] -= self.deltatau*(self.string[0]-self.avgconfig[0])
                #self.string[-1] -= self.deltatau*(self.string[-1]-self.avgconfig[-1])
