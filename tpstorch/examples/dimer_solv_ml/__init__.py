from tpstorch.ml import _ml
from . import _dimer_solv_ml

#Just something to pass the class
class MyMLEXPSampler(_dimer_solv_ml.DimerSolvEXP):
    pass

"""
class MyMLEXPStringSampler(_dimer_ml.DimerEXPString):
    pass
"""
class MyMLFTSSampler(_dimer_solv_ml.DimerSolvFTS):
    pass
