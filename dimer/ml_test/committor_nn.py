#Import necessarry tools from torch
import torch
import torch.nn as nn
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
        self.lin2 = nn.Linear(num_nodes, 1, bias=False)

    def forward(self, x):
        #X needs to be flattened
        x = x.view(-1,6)
        x = self.lin1(x)
        x = self.unit(x)
        x = self.lin2(x)
        return torch.sigmoid(x)

class CommittorNetDR(nn.Module):
    def __init__(self, num_nodes, boxsize, unit=torch.relu):
        super(CommittorNetDR, self).__init__()
        self.num_nodes = num_nodes
        self.unit = unit
        self.lin1 = nn.Linear(1, num_nodes, bias=True)
        self.lin2 = nn.Linear(num_nodes, 1, bias=False)
        self.boxsize = boxsize

    def forward(self, x):
        #Need to compute pair distance
        #By changing the view from flattened to 2 by x array
        x = x.view(-1,2,3)
        dx = x[:,0]-x[:,1]
        dx -= torch.round(dx/self.boxsize)*self.boxsize
        dx = torch.norm(dx,dim=1).view(-1,1)
        
        #Feed it to one hidden layer
        x = self.lin1(dx)
        x = self.unit(x)
        x = self.lin2(x)
        return torch.sigmoid(x)

class CommittorNetTwoHidden(nn.Module):
    def __init__(self, d, num_nodes, unit=torch.relu):
        super(CommittorNetTwoHidden, self).__init__()
        self.num_nodes = num_nodes
        self.d = d
        self.unit = unit
        self.lin1 = nn.Linear(d, num_nodes, bias=True)
        self.lin3 = nn.Linear(num_nodes, num_nodes, bias=True)
        self.lin2 = nn.Linear(num_nodes, 1, bias=False)

    def forward(self, x):

        x = self.lin1(x)
        x = self.unit(x)
        x = self.lin3(x)
        x = self.unit(x)
        x = self.lin2(x)
        return torch.sigmoid(x)

