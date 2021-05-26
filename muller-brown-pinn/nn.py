import torch
import torch.nn as nn

class CommittorNet(nn.Module):
    def __init__(self, d, num_nodes, beta, weight_pde, h_size=[200,200], unit=torch.relu):
        super(CommittorNet, self).__init__()
        self.num_nodes = num_nodes
        self.d = d
        self.beta_ = beta
        self.weight_pde = weight_pde
        self.unit = unit
        self.h_size = h_size
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(d, num_nodes, bias=True))
        for i in range(1, len(h_size)):
            self.fc.append(nn.Linear(h_size[i-1], h_size[i], bias=True))
        self.fc.append(nn.Linear(h_size[-1], 1, bias=False))
        self.thresh = torch.sigmoid

    def forward(self, x):
        #At the moment, x is flat. So if you want it to be 2x1 or 3x4 arrays, then you do it here!
        for i in range(len(self.h_size)-1):
            x = self.unit(self.fc[i](x))
        x = self.thresh(self.fc[-1](x))
        return x

    def energy_torch(self, z):
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

    def energy_torch_derivative(self, z):
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

    def loss_pde(self, x):
        q = self.forward(x)
        grad_energy = self.energy_torch_derivative(x)
        grad_q = torch.autograd.grad(q,x, grad_outputs=torch.ones_like(q), create_graph=True)[0]
        q_x = grad_q[:,0]
        q_y = grad_q[:,1]
        q_xx = torch.autograd.grad(q_x,x, grad_outputs=torch.ones_like(q_x), create_graph=True)[0][:,0]
        q_yy = torch.autograd.grad(q_y,x, grad_outputs=torch.ones_like(q_y), create_graph=True)[0][:,1]
        term_1 = self.beta_*(grad_energy[:,0]*q_x+grad_energy[:,1]*q_y)
        term_2 = q_xx+q_yy
        loss = term_2-term_1
        return self.weight_pde*(loss**2).mean()

    def loss_bc(self, x_b, u_b):
        q = self.forward(x_b)
        return ((q-u_b)**2).mean()

    def loss_cheat(self, x, q_exact):
        q = self.forward(x)
        return ((q-q_exact)**2).mean()
