import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

dist.init_process_group('mpi')
#Import necessarry tools from tpstorch 
from mullerbrown import CommittorNet
import numpy as np

#Import any other thing
import tqdm, sys

#Initialize neural net
committor = CommittorNet(d=2,num_nodes=200).to('cpu')
committor.load_state_dict(torch.load("simple_params_1"))

A= -10
a = -2
b = 0
c = -2
x = [0,1]
y = [0,1]

def V(xval,yval):
    val = 0
    for i in range(2):
        val += A*np.exp(a*(xval-x[i])**2+c*(yval-y[i])**2)
    return val

def q(xval,yval):
    qvals = np.zeros_like(xval)
    for i in range(nx):
        for j in range(ny):
            Array = np.array([xval[i,j],yval[i,j]]).astype(np.float32)
            Array = torch.from_numpy(Array)
            qvals[i,j] = committor(Array).item()
    return qvals
nx, ny = 10,10
X = np.linspace(-1.0, 2.0, nx)
Y = np.linspace(-1.0, 2.0, ny)

xv, yv = np.meshgrid(X, Y)
z = V(xv,yv)
h = plt.contourf(X,Y,z)
print(np.shape(z),np.shape(xv))
qvals = q(xv,yv)
CS = plt.contour(X, Y, qvals,levels=[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])
plt.colorbar()
plt.clabel(CS, fontsize=10, inline=1)#
plt.show()
