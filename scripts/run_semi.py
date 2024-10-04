#==============================================================================
#   Dependencies
#==============================================================================
from process_semi_data  import process_semi_data
from model_semi         import model_semi
from test_semi          import test_semi
from eval_semi          import eval_semi
from plot_semi          import plot_semi
#==============================================================================
def run_semi():
    """Run the HFTS semiconductor classification pipeline"""
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
        print(100*"*")
        print("Running the HFTS semiconductor classification pipeline")
        print(f"Classes = {c}")
        print(100*"*")
        print("Reading in the data.")
        print(100*"*")
        process_semi_data(c, noise=0.001)
        print(100*"*")
        print("Training the models.")
        print(100*"*")
        model_semi(c)
        print(100*"*")
        print("Testing the models.")
        print(100*"*")
        test_semi(c)
        print(100*"*")
        print("Generating the figures.")
        print(100*"*")
        plot_semi(c)
#==============================================================================
if __name__ == "__main__":
    run_semi()
#==============================================================================
