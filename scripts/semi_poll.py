#==============================================================================
#   Dependencies
#==============================================================================
from helpers import load_pickle_results, plot_cm
from numpy import array
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#==============================================================================
def semi_poll(classes=["microcontroller", "timer"]):
    """Poll the results of the HFTS semiconductor classification pipeline"""
    results = load_pickle_results(root="semi", classes=classes)
    predictions = array(results["predictions"]).T.tolist()
    labels = results["labels"][0]

    y_pred = []
    y_true = []
    for l, p in zip(labels, predictions):
        counts = Counter(p)
        print(f"{l} -> {counts.most_common(1)[0][0]}")
        y_pred.append(counts.most_common(1)[0][0])
        y_true.append(l)
    
    cm = confusion_matrix(y_true, y_pred)
    if classes==["microcontroller", "timer"]:
        dl = ["micro", "timer"]
    else: dl = classes
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dl)
    #disp.plot(cmap="Greys", colorbar=False, xticks_rotation="vertical")
    #plt.show()
    return disp


#==============================================================================
if __name__ == "__main__":
    classes = [
        ["microcontroller", "timer"],
        ["2210", "2230"],
        ["851t", "78j9", "93y8"],
        ["cc8z", "d855", "ckdf", "dont", "78j9", "93y8", "851t"],
    ]

    fig , axes = plt.subplots(2, 2, figsize=(10, 10))
    # Set the font size
    plt.rcParams.update({"font.size": 16})
    # Plot the confusion matrices
    plot_cm(semi_poll(classes=classes[0]),    axes[0][0])
    plot_cm(semi_poll(classes=classes[1]),   axes[0][1])
    plot_cm(semi_poll(classes=classes[2]),    axes[1][0])
    plot_cm(semi_poll(classes=classes[3]),   axes[1][1])
    # Remove the axis labels
    axes[0][0].set_xlabel("")
    axes[0][0].set_ylabel("")
    axes[0][1].set_xlabel("")
    axes[0][1].set_ylabel("")
    axes[1][0].set_xlabel("")
    axes[1][0].set_ylabel("")
    axes[1][1].set_xlabel("")
    axes[1][1].set_ylabel("")
    # Remove the yticks from the right figures
    #axes[1].set_yticks([],[])
    #axes[2].set_yticks([],[])
    #axes[3].set_yticks([],[])
    # Set some titles
    axes[0][0].set_title("(a)")
    axes[0][1].set_title("(b)")
    axes[1][0].set_title("(c)")
    axes[1][1].set_title("(d)")
    plt.tight_layout()
    plt.savefig("images/semi_poll.svg")


#==============================================================================