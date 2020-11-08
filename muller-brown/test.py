import torch
import tpstorch.fts
import mullerbrown as mb
a = mb.MySampler("param")
#These are just mock tensors
weights = torch.tensor([[0.0],[1.0]])
biases = torch.tensor([312.0,31212.0])
a.runSimulation(10**6,weights,biases)
