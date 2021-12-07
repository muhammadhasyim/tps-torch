#Import necessarry tools from torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn as nn

#Import necessarry tools from tpstorch 
#from brownian_ml import CommittorNet
from committor_nn import CommittorNet, CommittorNetDR
import numpy as np

#Import any other thing
import tqdm, sys

prefix = 'simple'
#Initialize neural net
#committor = CommittorNetDR(d=1,num_nodes=200).to('cpu')
committor = CommittorNetDR(num_nodes=2500, boxsize=10).to('cpu')
committor.load_state_dict(torch.load("{}_params_t_200_0".format(prefix)))

#Computing solution from neural network
#Reactant
r0 = 2**(1/6.0)
width = 0.5*r0
dist_init = r0-0.95*r0
start = torch.zeros((2,3))
start[0][2] = -0.5*dist_init
start[1][2] = 0.5*dist_init

#Product state
dist_init = r0+2*width+0.95*r0
end = torch.zeros((2,3))
end[0][2] = -0.5*dist_init
end[1][2] = 0.5*dist_init
def initializer(s):
    return (1-s)*start+s*end
s = torch.linspace(0,1,100)
x = []
y = []
boxsize = 10.0
for val in s:
    r = initializer(val)
    dr = r[1]-r[0]
    dr -= torch.round(dr/boxsize)*boxsize
    dr = torch.norm(dr)#.view(-1,1)
    x.append(dr)
    y.append(committor(r).item())

#Load exact solution
data = np.loadtxt('committor.txt')

import matplotlib.pyplot as plt
plt.figure(figsize=(7,4))
#Neural net solution vs. exact solution
plt.plot(x,y,'-')
plt.plot(data[:,0],data[:,1])

plt.show()
