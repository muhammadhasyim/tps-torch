import torch
from torch.optim import Optimizer, SGD
from torch.optim.optimizer import required
import torch.distributed as dist

#A heper function to project a set of vectors into a simplex
#Unashamedly copied from https://github.com/smatmo/ProjectionOntoSimplex/blob/master/project_simplex_pytorch.py
def project_simplex(v, z=1.0, axis=-1):
    """
    Implements the algorithm in Figure 1 of
    John Duchi, Shai Shalev-Shwartz, Yoram Singer, Tushar Chandra,
    "Efficient Projections onto the l1-Ball for Learning in High Dimensions", ICML 2008.
    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    This algorithm project vectors v onto the simplex w >= 0, \sum w_i = z.
    :param v: A torch tensor, will be interpreted as a collection of vectors.
    :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
    :param axis: Indicates the axis of v, which defines the vectors to be projected.
    :return: w: result of the projection
    """

    def _project_simplex_2d(v, z):
        """
        Helper function, assuming that all vectors are arranged in rows of v.
        :param v: NxD torch tensor; Duchi et al. algorithm is applied to each row in vecotrized form
        :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
        :return: w: result of the projection
        """
        with torch.no_grad():
            shape = v.shape
            if shape[1] == 1:
                w = v.clone().detach()
                w[:] = z
                return w

            mu = torch.sort(v, dim=1)[0]
            mu = torch.flip(mu, dims=(1,))
            cum_sum = torch.cumsum(mu, dim=1)
            j = torch.unsqueeze(torch.arange(1, shape[1] + 1, dtype=mu.dtype, device=mu.device), 0)
            rho = torch.sum(mu * j - cum_sum + z > 0.0, dim=1, keepdim=True) - 1.
            max_nn = cum_sum[torch.arange(shape[0],dtype=torch.long), rho[:, 0].type(torch.long)]
            theta = (torch.unsqueeze(max_nn, -1) - z) / (rho.type(max_nn.dtype) + 1)
            w = torch.clamp(v - theta, min=0.0)
            return w

    with torch.no_grad():
        shape = v.shape

        if len(shape) == 1:
            return _project_simplex_2d(torch.unsqueeze(v, 0), z)[0, :]
        else:
            axis = axis % len(shape)
            t_shape = tuple(range(axis)) + tuple(range(axis + 1, len(shape))) + (axis,)
            tt_shape = tuple(range(axis)) + (len(shape) - 1,) + tuple(range(axis, len(shape) - 1))
            v_t = v.permute(t_shape)
            v_t_shape = v_t.shape
            v_t_unroll = torch.reshape(v_t, (-1, v_t_shape[-1]))

            w_t = _project_simplex_2d(v_t_unroll, z)

            w_t_reroll = torch.reshape(w_t, v_t_shape)
            return w_t_reroll.permute(tt_shape)

#A helper function to compute the un-normalized reweighting factors
# k, index for reference
#fwd_meanwgtfactor, a 1D tensor of w_{l+1}/w_{l}, averaged over a mini-batch
#bwrd_meanwgtfactor, a 1D tensor of w_{l-1}/w_{l}, averaged over a mini-batch
def computeZl(k,fwd_meanwgtfactor,bwrd_meanwgtfactor):
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
