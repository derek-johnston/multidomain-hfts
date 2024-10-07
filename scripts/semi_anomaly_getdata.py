#==============================================================================
#   Dependencies
#==============================================================================
from helpers        import dump_pickle_adata
from os             import listdir
from pandas         import DataFrame, read_csv
from numpy.random   import seed
#==============================================================================
def semi_anomaly_getdata(classes=["microcontroller", "timer"]):
    """Perform anomaly detection on HFTS semiconductor data"""
    # Read-in the datasets
    datasets_n = []
    datasets_a = []
    for file in listdir("data/semiconductor/"):
        if classes[0] in file:
            print(f"Reading {file}...")
            dataset = read_csv(f"data/semiconductor/{file}", header=None)
            dataset["label"] = classes[0]
            datasets_n.append(dataset)
        if classes[1] in file:
            print(f"Reading {file}...")
            dataset = read_csv(f"data/semiconductor/{file}", header=None)
            dataset["label"] = classes[1]
            datasets_a.append(dataset)
    N = len(datasets_n)
    A = len(datasets_a[0:4])
    datasets = [DataFrame() for _ in range(64)]
    for j, data in enumerate(datasets_n):
        for i in range(64):
            print(f"({j+1}) Appending Nominal dataset {i+1}...")
            df = datasets[i]
            row = data.iloc[i]
            df = df._append(row, ignore_index=True)
            datasets[i] = df
    for data in datasets_a[0:4]:
        for i in range(64):
            print(f"({j+1}) Appending Anomalous dataset {i+1}...")
            df = datasets[i]
            row = data.iloc[i]
            df = df._append(row, ignore_index=True)
            datasets[i] = df
    seed(42)
    for i in range(64):
        datasets[i] = datasets[i].sample(frac=1, random_state=42).reset_index(drop=True)
    dump_pickle_adata(datasets, classes=classes)
#==============================================================================
if __name__ == "__main__":
    semi_anomaly_getdata()
#==============================================================================