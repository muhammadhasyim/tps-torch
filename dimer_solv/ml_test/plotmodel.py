#Import necessarry tools from torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn as nn

#Import necessarry tools from tpstorch 
#from brownian_ml import CommittorNet
from committor_nn import CommittorNet, CommittorNetBP, CommittorNetDR
import numpy as np

#Import any other thing
import tqdm, sys
Np = 32
prefix = 'simple'
#Initialize neural net
#committor = CommittorNetDR(d=1,num_nodes=200).to('cpu')
box = [14.736125994561544, 14.736125994561544, 14.736125994561544]
committor = CommittorNetBP(num_nodes=200, boxsize=box[0]).to('cpu')
committor.load_state_dict(torch.load("us_sl/{}_params_t_230_0".format(prefix)))

#Initialize neural net
def initializer(s):
    return (1-s)*start+s*end

#Initialization
r0 = 2**(1/6.0)
width =  0.5*r0

#Reactant
dist_init_start = r0
#Product state
dist_init_end = r0+2*width

#scale down/up the distance of one of the particle dimer
def CubicLattice(dist_init):
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

start = CubicLattice(dist_init_start)
end = CubicLattice(dist_init_end)

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
plt.figure(figsize=(5,5))
#Neural net solution vs. exact solution
plt.plot(x,y,'-')
plt.plot(data[:,0],data[:,1],'--')
plt.savefig('test.png')
plt.close()
#plt.show()
