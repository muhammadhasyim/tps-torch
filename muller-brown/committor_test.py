import torch
import numpy as np
import tpstorch.fts as fts
import mullerbrown as mb

# Basically I will read in averaged Voronoi cells, and then determine the committor at the Voronoi cells
committor_vals = np.zeros((32,))
committor_std = np.zeros((32,))

def myprod_checker(config):
    end = torch.tensor([[0.5,0.0]])
    radii = 0.2
    end_ = config-end
    end_ = end_.pow(2).sum()**0.5
    if ((end_ <= radii) or (config[1]<(config[0]+0.8))):
        return True
    else:
        return False

def myreact_checker(config):
    start = torch.tensor([[-0.5,1.5]])
    radii = 0.2
    start_ = config-start
    start_ = start_.pow(2).sum()**0.5
    if ((start_ <= radii) or (config[1]>(0.5*config[0]+1.5))):
        return True
    else:
        return False


for i in range(32):
    # Get config to test
    configs = np.genfromtxt(str(i)+"/config_"+str(i)+".xyz", skip_header=19000,usecols=(1,2))
    config_avg = np.mean(configs[1::2],axis=0)
    start = torch.from_numpy(config_avg)
    start = start.float()
    mb_sim = mb.MySampler("param_test",start, int(i), int(0))
    counts = []
    for j in range(1000):
        hitting = False
        mb_sim.setConfig(start)
        while hitting is False:
            mb_sim.runStep()
            config_ = mb_sim.getConfig()
            if myprod_checker(config_) is True:
                counts.append(1.0)
                hitting = True
            if myreact_checker(config_) is True:
                counts.append(0.0)
                hitting = True
    # Now compute the committor
    counts = np.array(counts)
    mean_count = np.mean(counts)
    conf_count = 1.96*np.std(counts)/len(counts)**0.5
    committor_vals[i] = mean_count
    committor_std[i] = conf_count 
    print("{:.8E} {:.8E} {:.8E} {:.8E}".format(config_avg[0],config_avg[1],mean_count,conf_count))

