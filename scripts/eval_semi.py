#==============================================================================
#   Dependencies
#==============================================================================
import matplotlib.pyplot as plt
from helpers import load_pickle_results, load_pickle_tests
from numpy   import array, unique
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
#==============================================================================
def eval_semi(classes=["microcontroller", "timer"]):
    """Evaluate the results of the HFTS semiconductor classification pipeline"""
    # Load the results
    results     = load_pickle_results(root="semi", classes=classes)
    # Convert the results to arrays
    scores      = array(results["scores"])
    predictions = array(results["predictions"])
    labels      = array(results["labels"])
    y_true = labels.flatten()
    y_pred = predictions.flatten()
    print(100*"*")
    print("CLASSIFICATION REPORT")
    print(100*"*")
    print(classification_report(y_true, y_pred, target_names=unique(y_true)))
    print(100*"*")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique(y_true))
    disp.plot()
    plt.show()
#==============================================================================
if __name__ == "__main__":
    classes = [
        ["microcontroller", "timer"],
        ["2210", "2230"],
        ["cc8z", "d855"],
        ["ckdf", "dont"],
        ["cc8z", "d855", "ckdf", "dont"],
        ["555cm", "555im"],
        ["78j9", "93y8"],
        ["851t", "78j9", "93y8"],
        ["cc8z", "d855", "ckdf", "dont", "78j9", "93y8", "851t"],
    ]
    for c in classes:
        eval_semi(classes=c)
        break
#==============================================================================
