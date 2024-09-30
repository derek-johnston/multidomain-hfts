#==============================================================================
#   Dependencies
#==============================================================================
import matplotlib.pyplot        as plt
import pandas                   as pd
from helpers                    import get_data, gen_model, plot_cm
def eid_allconc():
    """Classify electrolytes at all concentrations"""
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
    #==========================================================================
    #   Compile the datasets
    #==========================================================================
    data = pd.concat([
        kcl_15um,   kcl_150um,      kcl_15mm,   kcl_150mm,
        kno3_15um,  kno3_150um,     kno3_15mm,  kno3_150mm,
        mgcl2_15um, mgcl2_150um,    mgcl2_15mm, mgcl2_150mm,
        nacl_15um,  nacl_150um,     nacl_15mm,  nacl_150mm,
        nano3_15um, nano3_150um,    nano3_15mm, nano3_150mm
    ]).sample(frac=1)
    #==========================================================================
    #   Generate the model
    #==========================================================================
    cm = gen_model(data)
    #==========================================================================
    #   Generate Confusion Matrix Plot
    #==========================================================================
    fig, axes = plt.subplots(1, 1, figsize=(3.25, 3.25))
    plt.rcParams.update({"font.size": 9})
    plot_cm(cm, axes)
    axes.set_xlabel("")
    axes.set_ylabel("")
    plt.tight_layout()
    plt.savefig("images/eid_allconc.svg")
#==============================================================================
if __name__ == "__main__": eid_allconc()