#Import necessarry tools from torch
import torch
import torch.nn as nn
from nn import CommittorNet
import numpy as np
from scipy.optimize import minimize

torch.manual_seed(0)
np.random.seed(0)


#Initialize neural net
hidden = [20,20,20,20,20,20,20,20]
committor = CommittorNet(d=2,num_nodes=20,beta=1.0,weight_pde=1.0, h_size=hidden).to('cpu')

def energy(z):
    A = np.array([-20,-10,-17,1.5])
    a = np.array([-1,-1,-6.5,0.7])
    b = np.array([0,0,11,0.6])
    c = np.array([-10,-10,-6.5,0.7])
    x_ = np.array([1,0,-0.5,-1])
    y_ = np.array([0,0.5,1.5,1])
    x = z[0]
    y = z[1]
    energy_ = np.zeros((x.shape))
    for i in range(len(A)):
        energy_ += A[i]*np.exp(a[i]*(x-x_[i])**2+b[i]*(x-x_[i])*(y-y_[i])+c[i]*(y-y_[i])**2)
    return energy_

def energy_torch(z):
    A = torch.tensor([-20,-10,-17,1.5], dtype=torch.float)
    a = torch.tensor([-1,-1,-6.5,0.7], dtype=torch.float)
    b = torch.tensor([0,0,11,0.6], dtype=torch.float)
    c = torch.tensor([-10,-10,-6.5,0.7], dtype=torch.float)
    x_ = torch.tensor([1,0,-0.5,-1], dtype=torch.float)
    y_ = torch.tensor([0,0.5,1.5,1], dtype=torch.float)
    energy_ = torch.zeros_like(z[:,0], dtype=torch.float)
    for i in range(len(A)):
        energy_ = energy_ + A[i]*torch.exp(a[i]*(z[:,0]-x_[i])**2+b[i]*(z[:,0]-x_[i])*(z[:,1]-y_[i])+c[i]*(z[:,1]-y_[i])**2)
    return energy_

def energy_torch_derivative(z):
    A = torch.tensor([-20,-10,-17,1.5], dtype=torch.float)
    a = torch.tensor([-1,-1,-6.5,0.7], dtype=torch.float)
    b = torch.tensor([0,0,11,0.6], dtype=torch.float)
    c = torch.tensor([-10,-10,-6.5,0.7], dtype=torch.float)
    x_ = torch.tensor([1,0,-0.5,-1], dtype=torch.float)
    y_ = torch.tensor([0,0.5,1.5,1], dtype=torch.float)
    energy_ = torch.zeros_like(z, dtype=torch.float)
    for i in range(len(A)):
        energy_[:,0] = energy_[:,0] + A[i]*(2*a[i]*(z[:,0]-x_[i])+b[i]*(z[:,1]-y_[i]))*torch.exp(a[i]*(z[:,0]-x_[i])**2+b[i]*(z[:,0]-x_[i])*(z[:,1]-y_[i])+c[i]*(z[:,1]-y_[i])**2)
        energy_[:,1] = energy_[:,1] + A[i]*(b[i]*(z[:,0]-x_[i])+2*c[i]*(z[:,1]-y_[i]))*torch.exp(a[i]*(z[:,0]-x_[i])**2+b[i]*(z[:,0]-x_[i])*(z[:,1]-y_[i])+c[i]*(z[:,1]-y_[i])**2)
    return energy_


#Reactant minima
x0_react = [-0.5,1.5]
res = minimize(energy, x0_react, method='Nelder-Mead', options={'gtol': 1e-12, 'disp': True})
x_react = res.x
print(x_react)
np.savetxt("react_min.txt", res.x)

#Product minima
x0_prod = [0.5,0.0]
res = minimize(energy, x0_prod, method='Nelder-Mead', options={'gtol': 1e-12, 'disp': True})
x_prod = res.x
print(x_prod)
np.savetxt("prod_min.txt", res.x)

#Generate samples around minima to use as BC data
#n_boundary_samples = 100
#Actually lets be really brave and just try with the reactant and product minima
x_bc = np.hstack((x_react,x_prod))
u_bc = np.array([[0.0],[1.0]])

# load BC data into torch
x_bc = x_bc.reshape((2, 2))
x_bc = torch.tensor(x_bc, requires_grad=True, dtype=torch.float32)
u_bc = torch.tensor(u_bc, dtype=torch.float32)

r_start = np.array([[0.5,0.0],[-0.1,0.5],[-0.8,0.55],[-1.0,1.0],[-1.0,1.0],[0.5,0.0]])
r_end = np.array([[-0.1,0.5],[-0.8,0.55],[-1.0,1.0],[-0.2,1.8],[-1.3,0.7],[0.8,0.0]])

# Now generate a bunch of random points as a demonstration
x = np.zeros((6*10,2))
sigmas = np.array([[0.1],[0.1],[0.1],[0.1],[0.1],[0.1]])
for i in range(r_start.shape[0]):
    r_start_ = r_start[i,:]
    r_end_ = r_end[i,:]
    r_points = np.zeros((10,2))
    for j in range(10):
        r_points[j,:] = j/9*r_end_+(1-j/9)*r_start_
    r_points_2 = np.random.normal(loc=np.array([0.0,0.0]), scale=sigmas[i], size=(10,2))
    r_points += r_points_2
    x[int(i*10):int(i*10+10),:] = r_points

np.savetxt("x_points.txt", x)
q = np.genfromtxt("q_points.txt")

x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
q = torch.tensor(q, requires_grad=False, dtype=torch.float32)

# optimizer
optimizer = torch.optim.Adamax(committor.parameters(), lr=0.001)
losses = []
loss_fn = nn.MSELoss()
for i in range(10000):
    train_out_q = committor(x_bc)
    loss = loss_fn(train_out_q,u_bc)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(i)
    print(loss.detach().numpy())
    losses.append(loss.detach().numpy())

# optimizer.zero_grad()
# for j in range(6):
    # for i in range(10000):
        # train_out_q_bc = committor(x_bc)
        # loss_bc = loss_fn(train_out_q_bc,u_bc)
        # train_out_q = committor(x[(j*10):((j+1)*10),:])
        # loss_exact = loss_fn(train_out_q,q[(j*10):((j+1)*10)])
        # loss = loss_bc+loss_exact
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # print(i)
        # print(loss.detach().numpy())
        # losses.append(loss.detach().numpy())

# optimizer.zero_grad()
# for j in range(3):
    # for i in range(10000):
        # train_out_q_bc = committor(x_bc)
        # loss_bc = loss_fn(train_out_q_bc,u_bc)
        # train_out_q = committor(x[(j*20):((j+1)*20),:])
        # loss_exact = loss_fn(train_out_q,q[(j*20):((j+1)*20)])
        # loss = loss_bc+loss_exact
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # print(i)
        # print(loss.detach().numpy())
        # losses.append(loss.detach().numpy())

# optimizer.zero_grad()
# for j in range(2):
    # for i in range(10000):
        # train_out_q_bc = committor(x_bc)
        # loss_bc = loss_fn(train_out_q_bc,u_bc)
        # train_out_q = committor(x[(j*30):((j+1)*30),:])
        # loss_exact = loss_fn(train_out_q,q[(j*30):((j+1)*30)])
        # loss = loss_bc+loss_exact
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # print(i)
        # print(loss.detach().numpy())
        # losses.append(loss.detach().numpy())

optimizer.zero_grad()
for i in range(10000):
    train_out_q_bc = committor(x_bc)
    loss_bc = loss_fn(train_out_q_bc,u_bc)
    train_out_q = committor(x)
    loss_exact = loss_fn(train_out_q,q)
    loss = loss_bc+loss_exact
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(i)
    print(loss.detach().numpy())
    losses.append(loss.detach().numpy())

# batch_size = 5
# for j in range(1000):
    # print(j)
    # permutation = torch.randperm(q.size()[0])
    # for i in range(0, q.size()[0], batch_size):
        # optimizer.zero_grad()
        # indices = permutation[i:i+batch_size]
        # train_out_q_bc = committor(x_bc)
        # loss_bc = loss_fn(train_out_q_bc,u_bc)
        # train_out_q = committor(x[indices])
        # loss_exact = loss_fn(train_out_q,q[indices])
        # loss = loss_bc+loss_exact
        # loss.backward()
        # optimizer.step()
        # print(loss.detach().numpy())
        # losses.append(loss.detach().numpy())

# training
def train(epoch):
    committor.train()
    def closure():
        optimizer.zero_grad()
        #loss_pde = committor.loss_pde(x)
        #loss_bc = committor.loss_bc(x_bc, u_bc)
        loss_cheat = committor.loss_cheat(x, q)
        #loss = loss_pde + loss_bc + loss_cheat
        loss = loss_cheat
        loss.backward()
        return loss
    loss = optimizer.step(closure)
    loss_value = loss.item() if not isinstance(loss, float) else loss
    print(f'epoch {epoch}: loss {loss_value:.6f}')

# epochs = 1000
# committor.weight_pde = 0.0
# for epoch in range(epochs):
    # train(epoch)

# epochs = 1000
# committor.weight_pde = 0.0
# for epoch in range(epochs):
    # train(epoch)

torch.save(committor.state_dict(), "simple_params")
