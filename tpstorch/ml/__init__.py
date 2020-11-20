#Sooner or later, we'll implement a run function that encapsulates all of this
from . import _ml, optim, nn, data
from torch.distributed import distributed_c10d 

class MLSamplerEXP(_ml.MLSamplerEXP):
    def __init__(self, initial_config):
        super(MLSamplerEXP,self).__init__(initial_config, distributed_c10d._get_default_group())
