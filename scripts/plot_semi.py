#==============================================================================
#   Dependencies
#==============================================================================
from helpers import load_pickle_results
import matplotlib.pyplot as plt
from numpy.random import normal
from numpy import mean
#==============================================================================
def plot_semi(classes=["microcontroller", "timer"]):
    """Plot the results of the HFTS semiconductor classification pipeline"""
    # Load the results
    results = load_pickle_results(root="semi", classes=classes)
    means = results["means"]
    sdevs = results["sdevs"]
    
    diffs = [mean - sdev for mean, sdev in zip(means, sdevs)]

    synth_data = []
    for i in range(64):
        synth_data.append(normal(means[i], sdevs[i], 1000))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.rcParams.update({"font.size": 9})
    #ax.bar(range(len(means)), means, color="black", edgecolor="black", hatch="//", width=0.5)
    ax.scatter(range(len(means)), means, color="black", marker="s")
    ax.scatter(range(len(diffs)), diffs, color="dimgray", marker="_")
    for i, (m, diff) in enumerate(zip(means, diffs)):
        ax.vlines(x=i, ymin=diff, ymax=m, color="dimgray", linewidth=0.5)
    ax.hlines(y=mean(means), xmin=-1, xmax=64, color="dimgray", linestyle="--", linewidth=0.5)
    #ax.bar(range(len(diffs)), diffs, color="white", edgecolor="dimgray", width=0.5)
    ax.legend(["$\\mu$", "$\\mu - \\sigma$"], frameon=False, loc="upper left")
    ax.set_xlim(-1, 64)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("DUT Pin Permutation")
    ax.set_ylabel("Accuracy Score")
    plt.tight_layout()
    plt.savefig("images/semi.svg")

#==============================================================================
if __name__ == "__main__":
    plot_semi(classes=["cc8z", "d855", "ckdf", "dont", "78j9", "93y8", "851t"])
#==============================================================================