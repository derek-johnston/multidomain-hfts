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

    correct = [0 for _ in range(64)]
    for j in range(64):
        preds = predictions[j]
        print(len(preds))
        for i, label in enumerate(labels):
            p = preds[i]
            print(f"({i+1}) {label} -> {p}")
            if label == p:
                correct[j] += 1
    print(f"({len(correct)}) ({type(correct)}) {correct}")
    print(f"({len(scores)}) ({type(scores)}) {scores}")

    """
    for i, label in enumerate(labels):
        print(f"({i+1}) {label}")
        for j in range(64):
            print(f"{predictions[j]}")
            print(len(predictions[j]))
            return
    """
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
