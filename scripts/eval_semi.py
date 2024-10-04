#==============================================================================
#   Dependencies
#==============================================================================
from helpers import load_pickle_results, load_pickle_tests
from numpy   import array, unique
#==============================================================================
def eval_semi(classes=["microcontroller", "timer"]):
    """Evaluate the results of the HFTS semiconductor classification pipeline"""
    # Load the results
    results     = load_pickle_results(root="semi", classes=classes)
    t_scores    = load_pickle_tests(root="semi", classes=classes)
    # Convert the results to arrays
    scores      = array(results["scores"])
    predictions = array(results["predictions"])
    labels      = array(results["labels"])
    t_scores    = array(t_scores)
    correct = 0
    for prediction, label in zip(predictions, labels):
        # Find the most frequent string in the labels array
        unique_labels, counts = unique(label, return_counts=True)
        most_frequent_label = unique_labels[counts.argmax()]
        if label == most_frequent_label:
            correct += 1
    print(f"Correct = {correct} out of {len(labels)} | {100 * correct / len(labels)}%")
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
        plot_semi(classes=c)
#==============================================================================
