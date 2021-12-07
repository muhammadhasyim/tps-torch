from tpstorch.ml import _ml
from . import _dimer_ml

#Just something to pass the class
class MyMLEXPSampler(_dimer_ml.DimerEXP):
    pass

class MyMLEXPStringSampler(_dimer_ml.DimerEXPString):
    pass

class MyMLFTSSampler(_dimer_ml.DimerFTS):
    pass
