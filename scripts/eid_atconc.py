#==============================================================================
#   Dependencies
#==============================================================================
import matplotlib.pyplot    as plt
import pandas               as pd
from helpers                import get_data
#==============================================================================
def eid_atconc():
    """Classify electrolytes at specific concentrations"""
    #==========================================================================
    #   Import 15um dataset.
    #==========================================================================
    root = "data/electrolyte/electrolyte_15um_"
    kcl_15um    = get_data(f"{root}kcl.csv",    label="kcl")
    kno3_15um   = get_data(f"{root}kno3.csv",   label="kno3")
    mgcl2_15um  = get_data(f"{root}mgcl2.csv",  label="mgcl2")
    nacl_15um   = get_data(f"{root}nacl.csv",   label="nacl")
    nano3_15um  = get_data(f"{root}nano3.csv",  label="nano3")
    #==========================================================================
    #   Import 150um dataset
    #==========================================================================
    root = "data/electrolyte/electrolyte_150um_"
    kcl_150um    = get_data(f"{root}kcl.csv",    label="kcl")
    kno3_150um   = get_data(f"{root}kno3.csv",   label="kno3")
    mgcl2_150um  = get_data(f"{root}mgcl2.csv",  label="mgcl2")
    nacl_150um   = get_data(f"{root}nacl.csv",   label="nacl")
    nano3_150um  = get_data(f"{root}nano3.csv",  label="nano3")
    #==========================================================================
    #   Import 15mm dataset.
    #==========================================================================
    root = "data/electrolyte/electrolyte_15mm_"
    kcl_15mm    = get_data(f"{root}kcl.csv",    label="kcl")
    kno3_15mm   = get_data(f"{root}kno3.csv",   label="kno3")
    mgcl2_15mm  = get_data(f"{root}mgcl2.csv",  label="mgcl2")
    nacl_15mm   = get_data(f"{root}nacl.csv",   label="nacl")
    nano3_15mm  = get_data(f"{root}nano3.csv",  label="nano3")
    #==========================================================================
    #   Import 150mm dataset
    #==========================================================================
    root = "data/electrolyte/electrolyte_150mm_"
    kcl_150mm    = get_data(f"{root}kcl.csv",    label="kcl")
    kno3_150mm   = get_data(f"{root}kno3.csv",   label="kno3")
    mgcl2_150mm  = get_data(f"{root}mgcl2.csv",  label="mgcl2")
    nacl_150mm   = get_data(f"{root}nacl.csv",   label="nacl")
    nano3_150mm  = get_data(f"{root}nano3.csv",  label="nano3")
#==============================================================================
if __name__ == "__main__": eid_atconc()