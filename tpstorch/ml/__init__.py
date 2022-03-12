from . import _ml, optim, nn, data
from tpstorch import _mpi_group

#Inherit the C++ compiled sampler class, and add the MPI process group by default
class MLSamplerEXP(_ml.MLSamplerEXP):
    def __init__(self, initial_config, world_size, rank):
        super(MLSamplerEXP,self).__init__(initial_config, world_size, rank)

#Inherit the C++ compiled sampler class, and add the MPI process group by default
class MLSamplerEXPString(_ml.MLSamplerEXPString):
    def __init__(self, initial_config, world_size, rank):
        super(MLSamplerEXPString,self).__init__(initial_config, world_size, rank)

#Inherit the C++ compiled sampler class, and add the MPI process group by default
class MLSamplerEMUSString(_ml.MLSamplerEMUSString):
    def __init__(self, initial_config):
        super(MLSamplerEMUSString,self).__init__(initial_config, _mpi_group)

#Inherit the C++ compiled sampler class, and add the MPI process group by default
class MLSamplerFTS(_ml.MLSamplerFTS):
    def __init__(self, initial_config, world_size, rank):
        super(MLSamplerFTS,self).__init__(initial_config, world_size, rank)
