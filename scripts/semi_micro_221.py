#==============================================================================
#   Dependencies
#==============================================================================
import  matplotlib.pyplot       as plt
from helpers                    import semi_classify
#==============================================================================
def semi_device():
    """Classify devices based on device type"""
    scores = semi_classify("2230", "ckdf", "dont")
    print(scores)
    
#==============================================================================
if __name__ == "__main__": semi_device()