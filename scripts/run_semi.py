#==============================================================================
#   Dependencies
#==============================================================================
from process_semi_data  import process_semi_data
from model_semi         import model_semi
from test_semi          import test_semi
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
        process_semi_data(c)
        model_semi(c)
        test_semi(c)
#==============================================================================
if __name__ == "__main__":
    run_semi()
#==============================================================================
