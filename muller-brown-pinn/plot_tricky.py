import numpy as np
import matplotlib.pyplot as plt
import torch

#Import necessarry tools from tpstorch 
from nn import CommittorNet
import numpy as np

#Import any other thing
import tqdm, sys

#Initialize neural net
committor = CommittorNet(d=2,num_nodes=200,beta=1).to('cpu')
committor.load_state_dict(torch.load("simple_params"))

A = np.array([-20,-10,-17,1.5])
a = np.array([-1,-1,-6.5,0.7])
b = np.array([0,0,11,0.6])
c = np.array([-10,-10,-6.5,0.7])
x_ = np.array([1,0,-0.5,-1])
y_ = np.array([0,0.5,1.5,1])

def energy(x,y,A,a,b,c,x_,y_):
    energy_ = np.zeros((x.shape))
    for i in range(len(A)):
        energy_ += A[i]*np.exp(a[i]*(x-x_[i])**2+b[i]*(x-x_[i])*(y-y_[i])+c[i]*(y-y_[i])**2)
    return energy_

def q(xval,yval):
    qvals = np.zeros_like(xval)
    for i in range(nx):
        for j in range(ny):
            Array = np.array([xval[i,j],yval[i,j]]).astype(np.float32)
            Array = torch.from_numpy(Array)
            qvals[i,j] = committor(Array).item()
    return qvals
nx, ny = 100,100
X = np.linspace(-2.0, 1.5, nx)
Y = np.linspace(-1.0, 2.5, ny)
print(X.shape)

xv, yv = np.meshgrid(X, Y)
z = energy(xv,yv,A,a,b,c,x_,y_)
h = plt.contourf(X,Y,z,levels=[-15+i for i in range(16)])
print(np.shape(z),np.shape(xv))
qvals = q(xv,yv)
CS = plt.contour(X, Y, qvals,levels=[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])
plt.colorbar()
plt.clabel(CS, fontsize=10, inline=1)#
plt.show()
