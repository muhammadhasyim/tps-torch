#Import necessarry tools from torch
import torch
import torch.nn as nn
import numpy as np

#Import any other thing
import tqdm, sys

# SchNet imports
from typing import Optional

import os
import warnings
import os.path as osp
from math import pi as PI

import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear, ModuleList
from torch_scatter import scatter
from torch_geometric.data.makedirs import makedirs
from torch_geometric.data import download_url, extract_zip, Dataset
from torch_geometric.nn import radius_graph, MessagePassing

#Initialize neural net
def initializer(s, start, end):
    return (1-s)*start+s*end

def CubicLattice(dist_init, box, Np):
    state = torch.zeros(Np, 3);
    num_spacing = np.ceil(Np**(1/3.0))
    spacing_x = box[0]/num_spacing;
    spacing_y = box[1]/num_spacing;
    spacing_z = box[2]/num_spacing;
    count = 0;
    id_x = 0;
    id_y = 0;
    id_z = 0;
    while Np > count:
        state[int(id_z+id_y*num_spacing+id_x*num_spacing*num_spacing)][0] = spacing_x*id_x-0.5*box[0];
        state[int(id_z+id_y*num_spacing+id_x*num_spacing*num_spacing)][1] = spacing_y*id_y-0.5*box[1];
        state[int(id_z+id_y*num_spacing+id_x*num_spacing*num_spacing)][2] = spacing_z*id_z-0.5*box[2];
        count += 1;
        id_z += 1;
        if(id_z==num_spacing):
            id_z = 0;
            id_y += 1;
        if(id_y==num_spacing):
            id_y = 0;
            id_x += 1;

    #Compute the pair distance
    dx = (state[0]-state[1])
    dx = dx-torch.round(dx/box[0])*box[0]
    
    #Re-compute one of the coordinates and shift to origin
    state[0] = dx/torch.norm(dx)*dist_init+state[1] 
    
    x_com = 0.5*(state[0]+state[1])
    for i in range(Np):
        state[i] -= x_com
        state[i] -= torch.round(state[i]/box[0])*box[0]
    return state;

def initializeConfig(s, r0, width, boxsize, Np):
    #Reactant
    dist_init_start = r0
    #Product state
    dist_init_end = r0+2*width
    start = CubicLattice(dist_init_start, boxsize, Np)
    end = CubicLattice(dist_init_end, boxsize, Np)
    return start, end, initializer(s, start, end)

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
        x = x.view(-1,32,3)
        dx = x[:,0]-x[:,1]
        dx -= torch.round(dx/self.boxsize)*self.boxsize
        dx = torch.norm(dx,dim=1).view(-1,1)
        
        #Feed it to one hidden layer
        x = self.lin1(dx)
        x = self.unit(x)
        x = self.lin2(x)
        return torch.sigmoid(x)

class CommittorNetBP(nn.Module):
    def __init__(self, num_nodes, boxsize, Np, rc, sigma, unit=torch.relu):
        super(CommittorNetBP, self).__init__()
        self.num_nodes = num_nodes
        self.unit = unit
        self.Np = Np
        self.rc = rc
        self.factor = 1/(sigma**2)
        self.lin1 = nn.Linear(Np, num_nodes, bias=True)
        self.lin2 = nn.Linear(num_nodes, 1, bias=False)
        self.boxsize = boxsize
    
    def forward(self, x):
        PI = 3.1415927410125732
        x = x.view(-1,self.Np,3)
        #Create input array with shape batch_size x # of particles
        inputt = torch.zeros((x.shape[0],self.Np))
        count = 0
        for i in range(self.Np):
            for j in range(self.Np):
                #Compute pairwise distance
                if i != j:
                    dx = x[:,j]-x[:,i]
                    dx -= torch.round(dx/self.boxsize)*self.boxsize
                    dx = torch.norm(dx,dim=1)#.view(-1,1)
                    #Compute inputt per sample in batch
                    for k, val in enumerate(dx):
                        if val < self.rc:
                            inputt[k,i] += torch.exp(-self.factor*val**2)*0.5*(torch.cos(PI*val/self.rc)+1)
        #Feed it to one hidden layer
        x = self.lin1(inputt)
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

class SchNet(torch.nn.Module):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    .. note::

        For an example of using a pretrained SchNet variant, see
        `examples/qm9_pretrained_schnet.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        qm9_pretrained_schnet.py>`_.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """

    url = 'http://www.quantum-machine.org/datasets/trained_schnet_models.zip'

    def __init__(self, hidden_channels: int = 128, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50,
                 cutoff: float = 10.0, max_num_neighbors: int = 32,
                 readout: str = 'add', dipole: bool = False,
                 mean: Optional[float] = None, std: Optional[float] = None,
                 atomref: Optional[torch.Tensor] = None, boxsize: float = 5.0,
                 Np: int = 32, dim: int = 3):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.readout = readout
        self.dipole = dipole
        self.readout = 'add' if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None
        self.boxsize = boxsize
        self.Np = Np
        self.dim = dim

        atomic_mass = torch.from_numpy(np.array([1,2]))
        self.register_buffer('atomic_mass', atomic_mass)

        self.embedding = Embedding(2, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)


    def forward(self, pos):
        """"""
        pos = pos.view(-1,self.dim)
        total_positions = pos.size(dim=0)
        num_configs = total_positions//self.Np
        z = torch.zeros((total_positions,1), dtype=torch.int)
        z = z.view(-1,self.Np)
        z[:,0] = 1
        z[:,1] = 1
        z = z.view(-1)
        batch = torch.zeros(int(num_configs*self.Np), dtype=torch.int64)
        for i in range(num_configs):
            batch[(self.Np*i):(self.Np*(i+1))] = i
        
        h = self.embedding(z)

        edge_index = radius_graph(pos, r=2*self.boxsize, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        dx = pos[row] - pos[col]
        dx = dx-torch.round(dx/self.boxsize)*self.boxsize
        edge_weight = (dx).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            h = h * (pos - c.index_select(0, batch))

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = scatter(h, batch, dim=0, reduce=self.readout)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return torch.sigmoid(out)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')



class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        C *= (edge_weight < self.cutoff).float()
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
