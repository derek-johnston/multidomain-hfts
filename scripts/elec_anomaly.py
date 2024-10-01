#==============================================================================
#   Dependencies
#==============================================================================
import  matplotlib.pyplot   as plt
from    numpy               import where, sum, invert
from    pandas              import concat
from    helpers             import get_data, detect_anomalies
#==============================================================================
def elec_anomaly():
    """Detect anomalous samples in electrolyte data"""
    #==========================================================================
    #   Import the 150mm datasets
    #==========================================================================
    noise = False
    root = "data/electrolyte/electrolyte_150mm_"
    kcl_150mm    = get_data(f"{root}kcl.csv",   label="kcl",    noise=noise)
    nacl_150mm   = get_data(f"{root}nacl.csv",  label="nacl",   noise=noise)
    #==========================================================================
    #   combine the dataset with anomalies
    #==========================================================================
    data = concat([
        kcl_150mm.sample(frac=0.99),
        nacl_150mm.sample(frac=0.01)
    ]).sample(frac=1)
    print(data.shape)
    #==========================================================================
    #   Perform anomaly detection
    #==========================================================================
    results = detect_anomalies(data, threshold=0.002)
    #==========================================================================
    #   Plot the results
    #==========================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plt.rcParams.update({"font.size": 9})
    # Confusion Matrix
    cm = [[results["n_tp"], results["n_fp"]], [results["n_fn"], results["n_tn"]]]
    axes[1].imshow(cm, cmap="Greys")
    for i in range(2):
        for j in range(2):
            if cm[i][j] > 50: color="white"
            else: color="black"
            axes[1].annotate(f"{str(cm[i][j])}%", xy=(j, i), ha="center", va="center", color=color, fontsize=16)
    axes[1].set_title("(b)")
    axes[1].set_yticks([0, 1], ["Normal", "Anomaly"])
    axes[1].set_xticks([0, 1], ["Normal", "Anomaly"])
    axes[1].set_ylabel("Predicted")
    axes[1].set_xlabel("Actual")
    # Scatter Plot
    color_true = "black"
    color_false = "dimgray"
    for r, i in zip(results["t_pos"], range(len(results["t_pos"]))):
        if i == 0: axes[0].scatter(r[0], r[1], color=color_true, marker="*", label="True Positive")
        else: axes[0].scatter(r[0], r[1], color=color_true, marker="*")
    for r, i in zip(results["t_neg"], range(len(results["t_neg"]))):
        if i == 0: axes[0].scatter(r[0], r[1], color=color_false, marker="*", label="True Negative")
        else: axes[0].scatter(r[0], r[1], color=color_false, marker="*")
    for r, i in zip(results["f_pos"], range(len(results["f_pos"]))):
        if i == 0: axes[0].scatter(r[0], r[1], color=color_true, marker="x", label="False Positive")
        else: axes[0].scatter(r[0], r[1], color=color_true, marker="x")
    for r, i in zip(results["f_neg"], range(len(results["f_neg"]))):
        if i == 0: axes[0].scatter(r[0], r[1], color=color_false, marker="x", label="False Negative")
        else: axes[0].scatter(r[0], r[1], color=color_false, marker="x")
    
    axes[0].legend(frameon=False)
    axes[0].set_ylabel("Average Distance")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_xlim([0, 1000])
    axes[0].set_ylim([0.0009, 0.0040])
    axes[0].set_title("(a)")
    plt.tight_layout()
    plt.savefig("images/elec_anomaly.svg")
    
#==============================================================================
if __name__ == "__main__": elec_anomaly()