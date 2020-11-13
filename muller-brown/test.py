import torch
import tpstorch.fts
import mullerbrown as mb
a = mb.MySampler("param")

#These are just mock tensors
#Intepreat the MD potential as a 2D system for one particle
left_weight = torch.tensor([[0.0,1.0]])
left_bias = torch.tensor(1)
right_weight = torch.tensor([[-1.0,1.0]])
right_bias = torch.tensor(2)
a.runSimulation(10**6,left_weight,right_weight,left_bias,right_bias)
