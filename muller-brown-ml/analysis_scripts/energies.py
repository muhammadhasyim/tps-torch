import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch.autograd import grad

#Import necessarry tools from tpstorch 
from mb_ml import CommittorNet
import numpy as np

#Import any other thing
import tqdm, sys

#Initialize neural net
committor = CommittorNet(d=2,num_nodes=200).to('cpu')

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

from scipy.integrate import simps
n_struct = 100
points_x = np.linspace(-1.75,1.25,n_struct)
points_y = np.linspace(-0.5,2.25,n_struct)
xx, yy = np.meshgrid(points_x,points_y)
#Now evaluate gradient stuff
energies = energy(xx,yy,A,a,b,c,x_,y_)
string_list = []
for i in range(0,20000,4):
    string_list.append("simple_params_t_"+str(i)+"_0")

values = np.genfromtxt("../../analysis_scripts/values_of_interest.txt")
q_fem = np.genfromtxt("../../analysis_scripts/zz_structured.txt")
indices = np.nonzero(values)

integral_eq = np.zeros((len(string_list),))
q_eq = np.zeros((len(string_list),))
f = open("energies_data.txt", 'a+')
f2 = open("q_int_data.txt", 'a+')
for frame in range(len(string_list)):
    zz = np.zeros_like(xx)
    committor.load_state_dict(torch.load(string_list[frame]))
    grad_2_zz = np.zeros_like(xx)
    for i in range(n_struct):
        for j in range(n_struct):
            test = torch.tensor([xx[i][j],yy[i][j]], dtype=torch.float32, requires_grad=True)
            test_y = committor(test)
            dy_dx = grad(outputs=test_y, inputs=test)
            zz[i][j] = test_y.item()
            grad_2_zz_ = dy_dx[0][0]**2+dy_dx[0][1]**2
            grad_2_zz[i][j] = grad_2_zz_.item()

    integral_eq[frame] = simps(simps(grad_2_zz*np.exp(-1.0*energies),points_y),points_x)
    f.write("{:.5e}\n".format(integral_eq[frame]))

    q_metric = values[indices]*np.abs(qvals[indices]-q_fem[indices])
    q_int = np.array((np.mean(q_metric),np.std(q_metric)/len(q_metric)**0.5))
    q_eq[frame] = q_int
    f2.write("{:.5e}\n".format(q_int))

np.savetxt("energies_time.txt", energies)
np.savetxt("q_time.txt", energies)
