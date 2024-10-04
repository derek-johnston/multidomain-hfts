#==============================================================================
#   Dependencies
#==============================================================================
from helpers import load_pickle_results, load_pickle_tests
#==============================================================================
def plot_semi(classes=["microcontroller", "timer"]):
    """Plot the results of the HFTS semiconductor classification pipeline"""
    # Load the results
    results = load_pickle_results(root="semi", classes=classes)
    t_scores = load_pickle_tests(root="semi", classes=classes)
    predictions = results["predictions"]
    labels = results["labels"]
    scores = results["scores"]
    N = len(predictions[0])
    accuracy = [0 for _ in range(64)]
    print(100*"*")
    print(len(labels))
    print(labels)
    print(100*"*")
    print(len(predictions))
    print(100*"*")
    print(len(predictions[0]))
    print(100*"*")
    for label, prediction in zip(labels, predictions):
        for i in range(64)
#==============================================================================
if __name__ == "__main__":
    plot_semi()
#==============================================================================