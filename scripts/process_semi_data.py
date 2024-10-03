#==============================================================================
#   Dependencies
#==============================================================================
from os     import listdir
from pandas import read_csv, DataFrame
from pickle import dump
#==============================================================================
def process_semi_data(classes=["microcontroller", "timer"]):
    """Read-in, process, and store an HFTS semiconductor dataset"""
    datasets = [DataFrame() for _ in range(64)]
    for c in classes:
        files = listdir(f"data/semiconductor")
        for f in files:
            if c in f:
                print(f"Reading in {f}")
                data = read_csv(f"data/semiconductor/{f}", header=None)
                data["label"] = c
                for i in range(64):
                    df = datasets[i]
                    row = data.iloc[i]
                    df = df._append(row, ignore_index=True)
                    datasets[i] = df
    # Shuffle the DataFrames
    for i in range(64): 
        datasets[i] = datasets[i].sample(frac=1).reset_index(drop=True)
    # Save the list of DataFrams as a pickle file
    p_filename = f"semi_d"
    for c in classes: 
        p_filename += f"_{c}"
    p_filename += ".pkl"
    with open(f"pickles/{p_filename}", "wb") as file:
        dump(datasets, file)
#==============================================================================
if __name__ == "__main__":
    process_semi_data(classes=["cc8z", "d855"])
#==============================================================================
