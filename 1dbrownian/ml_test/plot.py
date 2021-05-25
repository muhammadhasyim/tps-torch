#Import necessarry tools from torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.ticker import AutoMinorLocator
mpl.rcParams['text.usetex'] = True
params= {'text.latex.preamble' : [r'\usepackage{bm}',r'\usepackage{mathtools,amsmath}']}
mpl.rcParams.update(params)

#Import necessarry tools from tpstorch 
from brownian_ml import CommittorNet
from brownian_ml import CommittorFTSNet
import numpy as np

#Import any other thing
import tqdm, sys

#prefix = 'vanilla_highT'
prefix = 'fts_highT'
#prefix = 'cl_highT'

#Computing solution from neural network
start = torch.tensor([[-1.0]])
end = torch.tensor([[1.0]])
def initializer(s):
    return (1-s)*start+s*end

size = 11

#Initialize neural net
if prefix == 'fts_highT':
    committor = CommittorFTSNet(d=1,start=start.flatten(),end=end.flatten(),num_nodes=200, fts_size=size).to('cpu')
else:
    committor = CommittorNet(d=1,num_nodes=200).to('cpu')

committor.load_state_dict(torch.load("{}_params_1".format(prefix)))

data = np.loadtxt("{}_bp_1.txt".format(prefix))
kT = data[-1,1]

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

plt.figure(figsize=(6,3))
#Neural net solution vs. exact solution
plt.subplot(121)
plt.plot(x,y,label='NN')
plt.plot(newx,yexact,label='Exact')
plt.xlabel('$x$',fontsize=14)
plt.ylabel('$q(x)$',fontsize=14)
plt.legend(loc=0)

#The loss function over iterations
plt.subplot(122)
data = np.loadtxt("{}_loss.txt".format(prefix))
index = data[:,1] > 0
plt.semilogy(data[index,0],data[index,1],label='$\\frac{1}{2}| \\nabla q(x)|^2$')
plt.legend(loc=0)
plt.savefig('solution_{}.png'.format(prefix),dpi=300)
print(kT)

#Plotting Histogram of particle trajectories
plt.figure(figsize=(6,2))
for i in range(size):
    data = np.loadtxt("{}_bp_{}.txt".format(prefix,i+1))
    #plt.hist(data[:,0],bins='auto',histtype='step')
    plt.plot(data[:,0],'-')#,bins='auto',histtype='step')#x,y)
plt.savefig('hist_{}.png'.format(prefix),dpi=300)

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
