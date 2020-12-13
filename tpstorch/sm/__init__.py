from . import _sm

#Inherit the C++ compiled sampler class, and add the MPI process group by default
class Sampler(_sm.Sampler):
    def __init__(self):
        super(Sampler,self).__init__()
