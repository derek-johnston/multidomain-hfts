#==============================================================================
#   Dependencies
#==============================================================================
from pandas import read_csv
#==============================================================================
def get_data(path, label=""):
    """Read-in and parse an HFTS dataset."""
    data = read_csv(path)
    data = data.drop("Time", axis=1)
    data = data.astype(complex)
    data = data.apply(abs)
    if label != "":
        data["Label"] = label
    return data
#==============================================================================