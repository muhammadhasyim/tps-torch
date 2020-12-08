import torch
import tpstorch.fts
import mullerbrown as mb
a = mb.MySampler("param",torch.tensor([[0.0,1.0]]),0)

#These are just mock tensors
#Intepreat the MD potential as a 2D system for one particle
left_weight = torch.tensor([[1.0,-1.0]])
left_bias = torch.tensor(0.0)
right_weight = torch.tensor([[1.0,-1.0]])
right_bias = torch.tensor(2.0)
a.runSimulation(10**6,left_weight,right_weight,left_bias,right_bias)
