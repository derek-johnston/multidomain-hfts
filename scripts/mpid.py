#==============================================================================
#   Dependencies
#==============================================================================
import matplotlib.pyplot        as plt
from pandas                     import concat
from helpers                    import get_data, gen_model, plot_cm
#==============================================================================
def mpid():
    """Classify microparticles"""
    #==========================================================================
    #   Import the datasets.
    #==========================================================================
    root        = "data/microparticle/microparticle_amidine_"
    amidine1    = get_data(f"{root}01.csv", label="amidine")
    amidine2    = get_data(f"{root}02.csv", label="amidine")
    amidine3    = get_data(f"{root}03.csv", label="amidine")
    root        = "data/microparticle/microparticle_carbox_"
    carbox1     = get_data(f"{root}01.csv", label="carboxylate")
    carbox2     = get_data(f"{root}02.csv", label="carboxylate")
    carbox3     = get_data(f"{root}03.csv", label="carboxylate")
    #==========================================================================
    #   Compile the datasets.
    #==========================================================================
    data = concat([
        amidine1, amidine2, amidine3, carbox1, carbox2, carbox3
    ])
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
    plt.savefig("images/mpid.svg")
#==============================================================================
if __name__ == "__main__": mpid()