#Import necessarry tools from torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn as nn

#Import necessarry tools from tpstorch 
from brownian_ml import CommittorNet
import numpy as np

#Import any other thing
import tqdm, sys

prefix = 'highT'

#Initialize neural net
committor = CommittorNet(d=1,num_nodes=100).to('cpu')
committor.load_state_dict(torch.load("{}_params_1".format(prefix)))

data = np.loadtxt("{}_bp_1.txt".format(prefix))
kT = data[-1,1]

#Computing solution from neural network
start = torch.tensor([[-1.0]])
end = torch.tensor([[1.0]])
def initializer(s):
    return (1-s)*start+s*end
s = torch.linspace(0,1,200)
x = []
y = []
for val in s:
    x.append(initializer(val))
    y.append(committor(x[-1]).item())

#Computing exact solution
from scipy.integrate import quad
newx = torch.linspace(-1.0,1.0,100)
yexact = []
def integrand(x):
    return np.exp((1-x**2)**2/kT)
norm = quad(integrand,-1,1)[0]
def exact(x):
    return quad(integrand,-1,x)[0]/norm
for val in newx:
    yexact.append(exact(val.item()))

import matplotlib.pyplot as plt

plt.figure(figsize=(7,4))
#Neural net solution vs. exact solution
plt.subplot(121)
plt.plot(x,y)
plt.plot(newx,yexact)

#The loss function over iterations
plt.subplot(122)
data = np.loadtxt("{}_loss.txt".format(prefix))
plt.semilogy(data[:,0],data[:,1])
print(kT)

#Plotting Histogram of particle trajectories
plt.figure(figsize=(7,4))
for i in range(11):
    data = np.loadtxt("{}_bp_{}.txt".format(prefix,i+1))
    plt.hist(data[100::2,0],bins='auto',histtype='step')#x,y)

#Plot validation result
"""
plt.figure(figsize=(5,5))
for i in range(11):
    data = np.loadtxt("{}_validation_{}.txt".format(prefix,i+1))
    plt.errorbar(data[:,0],data[:,1],yerr=data[:,2],ls='None',color='k',marker='o')
x = np.linspace(0.3,0.7)
plt.plot(x,x,'--')
plt.ylim([0.3,0.7])
plt.xlim([0.3,0.7])
"""
plt.show()
