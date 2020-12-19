import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation

#This script simultaneously generate the plots for f(s) and compute scores 
fig, ax = plt.subplots(figsize=(9,4))

def V(x):
    return (1-x**2)**2

num_nodes = 11#8

bp = []
sgamma = []
kT = 0.4
xval = []
yval = []
yend = 0

plt.subplot(131)
for i in range(num_nodes):
    data = np.loadtxt("_bp_{}.txt".format(i+1))[10:,0]#[1:]#10]
    data = data[(data >= -1.25) & (data <= 1.25)]
    s = 0.5*(data+1)
    hist, bins = np.histogram(s)#100)
    center = (bins[:-1] + bins[1:]) / 2
    y =-np.log(hist)
    plt.plot(center,y,'-o')
plt.ylabel('$-\log[H_\\alpha]$',fontsize=15)
plt.xlabel('$s$',fontsize=15)
plt.xlim([0,1])

plt.subplot(132)
for i in range(num_nodes):
    data = np.loadtxt("_bp_{}.txt".format(i+1))[10:,0]#[1:]#10]
    data = data[(data >= -1.25) & (data <= 1.25)]
    s = 0.5*(data+1)
    #s = s[(s >= 0) & (s <= 1.0)]
    hist, bins = np.histogram(s)#100)
    center = (bins[:-1] + bins[1:]) / 2
    #print(hist)
    if i > 0:
        y =-kT*np.log(hist/hist[0])+np.polyval(z,center[0])
        plt.plot(center,y,'o')
    else:
        y = -kT*np.log(hist)
        plt.plot(center,y,'o')#-np.polyval(z,center[0]))
    yval.append(y)
    xval.append(center)
    z = np.polyfit(center,y,2)

yval = np.array(yval).flatten()
xval = np.array(xval).flatten()
z = np.polyfit(xval,yval,10)
x = np.linspace(-1,1)

plt.plot(xval,np.polyval(z,xval),color='k',label='Fitted Polynomial \n (10th order)',zorder=-1)#-np.polyval(z,0.0))
plt.legend(loc=0)
plt.ylabel('$G(s)$', fontsize=15)
plt.xlabel('$s$',fontsize=15)
plt.xlim([0,1])

#Computing exact solution and committor from the FTS method
from scipy.integrate import quad
newx = np.linspace(-1.0,1.0,100)
yexact = [] #stored exact solution
ynum = [] #stored numerical solution from the FTS method
def integrand(x):
    return np.exp((1-x**2)**2/kT)
def num_integrand(x):
    return np.exp(np.polyval(z,x)/kT)
norm = quad(integrand,-1,1)[0]
def exact(x):
    return quad(integrand,-1,x)[0]/norm
norm1 = quad(num_integrand,0,1)[0]
def numeric(x):
    return quad(num_integrand,0,0.5*(1+x))[0]/norm1
for val in newx:
    yexact.append(exact(val))
    ynum.append(numeric(val))

plt.subplot(133)
#np.savetxt('qdata.txt',ynum)
plt.plot(0.5*(newx+1),ynum)
plt.ylabel('$f(s)$',fontsize=15)
plt.xlabel('$s$',fontsize=15)
plt.xlim([0,1])
plt.ylim([0,1])
plt.tight_layout()
#plt.savefig('computingfs.pdf',dpi=300)

#Computing scores here
score = []
for i in range(11):
    #Loading up the batch of data generated from the FTS method
    tse_data = np.loadtxt('batch_fts_{}.txt'.format(i+1))
    emp_estimate = np.loadtxt('highT_fts_testvalidation_{}.txt'.format(i+1))[:,1]
    for j, xval in enumerate(tse_data):
        try:
            pred = numeric(xval)
            score.append(np.abs(pred-emp_estimate[j]))
        except:
            break
score = np.array(score)
print(1-np.mean(score),2.576*np.std(score)/len(score)**0.5)
