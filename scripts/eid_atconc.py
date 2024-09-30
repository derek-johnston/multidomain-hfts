#==============================================================================
#   Dependencies
#==============================================================================
import matplotlib.pyplot        as plt
import pandas                   as pd
from helpers                    import get_data, gen_model, plot_cm
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
    #==========================================================================
    #   # Compile the datasets
    #==========================================================================
    data_15um   = pd.concat([
        kcl_15um, kno3_15um, mgcl2_15um, nacl_15um, nano3_15um
    ]).sample(frac=1)
    data_150um  = pd.concat([
        kcl_150um, kno3_150um, mgcl2_150um, nacl_150um, nano3_150um
    ]).sample(frac=1)
    data_15mm   = pd.concat([
        kcl_15mm, kno3_15mm, mgcl2_15mm, nacl_15mm, nano3_15mm
    ]).sample(frac=1)
    data_150mm  = pd.concat([
        kcl_150mm, kno3_150mm, mgcl2_150mm, nacl_150mm, nano3_150mm
    ]).sample(frac=1)
    #==========================================================================
    #   Generate models
    #==========================================================================
    cm_15um     = gen_model(data_15um)
    cm_150um    = gen_model(data_150um)
    cm_15mm     = gen_model(data_15mm)
    cm_150mm    = gen_model(data_150mm)
    #==========================================================================
    #   Generate Confusion Matrix Plots
    #==========================================================================
    fig , axes = plt.subplots(1, 4, figsize=(10, 3.25))
    # Set the font size
    plt.rcParams.update({"font.size": 9})
    # Plot the confusion matrices
    plot_cm(cm_15um,    axes[0])
    plot_cm(cm_150um,   axes[1])
    plot_cm(cm_15mm,    axes[2])
    plot_cm(cm_150mm,   axes[3])
    # Remove the axis labels
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("")
    axes[3].set_xlabel("")
    axes[3].set_ylabel("")
    # Remove the yticks from the right figures
    axes[1].set_yticks([],[])
    axes[2].set_yticks([],[])
    axes[3].set_yticks([],[])
    # Set some titles
    axes[0].set_title("(a)")
    axes[1].set_title("(b)")
    axes[2].set_title("(c)")
    axes[3].set_title("(d)")
    plt.tight_layout()
    plt.savefig("images/eid_atconc.svg")
#==============================================================================
if __name__ == "__main__": eid_atconc()