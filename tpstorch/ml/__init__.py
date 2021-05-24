from . import _ml, optim, nn, data
from tpstorch import _mpi_group

#Inherit the C++ compiled sampler class, and add the MPI process group by default
class MLSamplerEXP(_ml.MLSamplerEXP):
    def __init__(self, initial_config):
        super(MLSamplerEXP,self).__init__(initial_config, _mpi_group)

#Inherit the C++ compiled sampler class, and add the MPI process group by default
class MLSamplerFTS(_ml.MLSamplerFTS):
    def __init__(self, initial_config):
        super(MLSamplerFTS,self).__init__(initial_config, _mpi_group)
