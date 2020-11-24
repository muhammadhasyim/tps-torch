#holds the loss function. And if we get the time, the custom neural net layer with the string path
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class CommittorLossEXP(_Loss):
    r"""Creates a loss function for the commitor based on the Backward Kolmogorov Equation.

    There are two components to the loss function. The main loss function takes samples of gradients of committor and 
    the 1/c(x) factor for reweighting. This should not be changed. The second component is the boundary condition loss function,
    which is not implemented by default, but generically, it will be a function of the committor, sampled configurations, and the 1/c(x) 
    factor for reweighting. 

    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None):
        super(CommittorLossEXP, self).__init__(size_average, reduce, 'none')
        self.loss = torch.zeros(1)
        self.bc_loss = torch.zeros(1)
    def compute_bc(self, committor, config):
        return NotImplementedError
    def compute_loss(self, gradients, invnormconstants):
        return 0.5*torch.sum(torch.mean(gradients*gradients*invnormconstants.view(-1,1),dim=0));
    def forward(self, gradients, committor, config, invnormconstants):
        self.loss = self.compute_loss(gradients, invnormconstants)
        self.bc_loss = self.compute_bc(committor, config, invnormconstants)
        return self.loss+self.bc_loss
