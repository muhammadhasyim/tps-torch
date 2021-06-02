import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch.autograd import grad

#Import necessarry tools from tpstorch 
from mb_fts import CommittorNet
from tpstorch.ml.nn import FTSLayer
import numpy as np

#Import any other thing
import tqdm, sys

#Initialize neural net
committor = CommittorNet(d=2,num_nodes=200).to('cpu')
committor.load_state_dict(torch.load("simple_params_1"))
start = torch.tensor([-0.5,1.5])
end = torch.tensor([0.6,0.08])
ftslayer = FTSLayer(react_config=start,prod_config=end,num_nodes=48).to('cpu')
ftslayer.load_state_dict(torch.load("simple_string_1"))
print(ftslayer.string)
ftslayer_np = ftslayer.string.cpu().detach().numpy()
print(ftslayer_np)

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
X = np.linspace(-1.75, 1.25, nx)
Y = np.linspace(-0.5, 2.25, ny)
print(X.shape)

xv, yv = np.meshgrid(X, Y)
z = energy(xv,yv,A,a,b,c,x_,y_)
h = plt.contourf(X,Y,z,levels=[-15+i for i in range(16)])
print(np.shape(z),np.shape(xv))
qvals = q(xv,yv)
CS = plt.contour(X, Y, qvals,levels=[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99], cmap='inferno')
plt.colorbar()
plt.clabel(CS, fontsize=10, inline=1)#
plt.plot(ftslayer_np[:,0], ftslayer_np[:,1], 'bo-')
plt.tick_params(axis='both', which='major', labelsize=9)
plt.tick_params(axis='both', which='minor', labelsize=9)
plt.savefig('energies.pdf', bbox_inches='tight')
#plt.show()
plt.close()

#Now evaluate gradient stuff
n_struct = 100
points_x = np.linspace(-1.75,1.25,n_struct)
points_y = np.linspace(-0.5,2.25,n_struct)
xx, yy = np.meshgrid(points_x,points_y)
zz = np.zeros_like(xx)
grad_2_zz = np.zeros_like(xx)
for i in range(n_struct):
    for j in range(n_struct):
        test = torch.tensor([xx[i][j],yy[i][j]], dtype=torch.float32, requires_grad=True)
        test_y = committor(test)
        dy_dx = grad(outputs=test_y, inputs=test)
        zz[i][j] = test_y.item()
        grad_2_zz_ = dy_dx[0][0]**2+dy_dx[0][1]**2
        grad_2_zz[i][j] = grad_2_zz_.item()

energies = energy(xx,yy,A,a,b,c,x_,y_)

from scipy.integrate import simps
energy_int = simps(simps(grad_2_zz*np.exp(-1.0*energies),points_y),points_x)
with open('energy.txt', 'w') as f:
    print(energy_int, file=f)

values = np.genfromtxt("../../analysis_scripts/values_of_interest.txt")
q_fem = np.genfromtxt("../../analysis_scripts/zz_structured.txt")
indices = np.nonzero(values)
q_metric = values[indices]*np.abs(qvals[indices]-q_fem[indices])
q_int = np.array((np.mean(q_metric),np.std(q_metric)/len(q_metric)**0.5))
np.savetxt("q_int.txt", q_int)
