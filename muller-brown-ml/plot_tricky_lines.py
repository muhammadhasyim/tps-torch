import numpy as np
import matplotlib.pyplot as plt

#Import any other thing
import tqdm, sys

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

nx, ny = 100,100
X = np.linspace(-2.0, 1.5, nx)
Y = np.linspace(-1.0, 2.5, ny)
print(X.shape)

# now plot lines on top of it
# Another line for the initial path
r_start = np.array([-1.2,0.9])
r_end = np.array([-0.5,0.5])
r_points = np.zeros((16,2))
for i in range(16):
    r_points[i,:] = i/15*r_end+(1-i/15)*r_start

xv, yv = np.meshgrid(X, Y)
z = energy(xv,yv,A,a,b,c,x_,y_)
h = plt.contourf(X,Y,z,levels=[-15+i for i in range(16)])
plt.plot(X,X+1)
plt.plot(X,0.5*(X-1)+2)
plt.plot(r_points[:,0],r_points[:,1])
print(np.shape(z),np.shape(xv))
plt.colorbar()
plt.show()
