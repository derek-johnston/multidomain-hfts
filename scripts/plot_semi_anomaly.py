#==============================================================================
#   Dependencies
#==============================================================================
from helpers                import load_pickle_aresults
import matplotlib.pyplot    as plt
#==============================================================================
def plot_semi_anomaly():
    """Plot the results of the semi-anomaly detection pipeline"""
    results = load_pickle_aresults()
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    plt.rcParams.update({"font.size": 9})
    ts = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"], ["(g)", "(h)", "(i)"]]
    for i in range(3):
        for j in range(3):
            Np = results[i][j]["tp"] + results[i][j]["fp"]
            Nn = results[i][j]["tn"] + results[i][j]["fn"]
            if Np == 0: Np = 1
            if Nn == 0: Nn = 1
            cm = [[results[i][j]["tp"]/Np, results[i][j]["fp"]/Np], [results[i][j]["fn"]/Nn, results[i][j]["tn"]/Nn]]
            cm_r = [[results[i][j]["tp"], results[i][j]["fp"]], [results[i][j]["fn"], results[i][j]["tn"]]]
            ax[i][j].imshow(cm, cmap="Greys")
            ax[i][j].set_title(f"{ts[i][j]}")
            ax[i][j].set_xticks([0, 1], ["Anomaly", "Normal"])
            ax[i][j].set_yticks([0, 1], ["Anomaly", "Normal"])
            # Annotate each cell with the numerical value
            for x in range(2):
                for y in range(2):
                    if cm[x][y] > 0.5: color="white"
                    else: color="black"
                    ax[i][j].text(y, x, f"{cm_r[x][y]}", ha="center", va="center", color=color, fontsize=16)
            #ax[i][j].tick_params(axis='y', labelrotation=90)
    plt.tight_layout()
    plt.savefig("images/semi_anomaly.svg")
#==============================================================================
if __name__ == "__main__":
    plot_semi_anomaly()
#==============================================================================