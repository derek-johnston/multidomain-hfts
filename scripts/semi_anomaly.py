#==============================================================================
#   Dependencies
#==============================================================================
from helpers        import detect_anomalies
from os             import listdir
from pandas         import DataFrame, read_csv
from numpy.random   import seed
#==============================================================================
def semi_anomaly(classes=["microcontroller", "timer"]):
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
    anomalies_idx = []
    thresholds = []
    averages = []
    for i in range(64):
        print(f"Detecting anomalies in dataset {i+1}...")
        data = datasets[i]
        labels = data["label"]
        threshold = 0.50
        best_threshold = 0.000
        max_avg = 0
        while threshold < 1.50:
            print(f"Threshold = {threshold}")
            results = detect_anomalies(data, real=classes[0], fake=classes[1], threshold=threshold)
            avg = (results["n_tp"] + results["n_tn"]) / 2
            print(f"{avg}")
            if avg > max_avg: 
                max_avg = (results["n_tp"] + results["n_tn"]) / 2
                best_threshold = threshold
            threshold = round(threshold + 0.01, 2)
        results = detect_anomalies(datasets[i], real=classes[0], fake=classes[1], threshold=best_threshold)
        thresholds.append(best_threshold)
        averages.append(max_avg)
        #print(f"Best threshold = {best_threshold} average = {max_avg}")
        #print(results["a_idx"])
        #print(labels[results["a_idx"][0]])
        anomalies_idx.append(results["a_idx"][0])
    for t, a in zip(thresholds, averages):
        print(f"Threshold = {t} Average = {a}")
#==============================================================================
if __name__ == "__main__":
    semi_anomaly()
#==============================================================================