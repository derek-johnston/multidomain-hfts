#==============================================================================
#   Dependencies
#==============================================================================
import  matplotlib.pyplot   as plt
from    numpy               import where, sum, invert
from    pandas              import concat
from    helpers             import get_data
#==============================================================================
def elec_anomaly():
    """Detect anomalous samples in electrolyte data"""
    #==========================================================================
    #   Import the 150mm datasets
    #==========================================================================
    noise = False
    root = "data/electrolyte/electrolyte_150mm_"
    kcl_150mm    = get_data(f"{root}kcl.csv",   label="kcl",    noise=noise)
    nacl_150mm   = get_data(f"{root}nacl.csv",  label="nacl",   noise=noise)
    #==========================================================================
    #   combine the dataset with anomalies
    #==========================================================================
    data = concat([
        kcl_150mm.sample(frac=0.9),
        nacl_150mm.sample(frac=0.1)
    ]).sample(frac=1)
    #==========================================================================
    #   Perform anomaly detection
    #==========================================================================
    X = data.drop("label", axis=1).to_numpy()
    y = data["label"].to_numpy()
    model = NearestNeighbors()
    model.fit(X)
    distances, _ = model.kneighbors(X)
    plt.plot(distances.mean(axis=1))
    anomaly_idxs = where(distances.mean(axis=1) > 0.0019)
    nominal_idxs = where(distances.mean(axis=1) <= 0.0019)
    anomalies = data.iloc[anomaly_idxs]
    nominals = data.iloc[nominal_idxs]
    indexes = anomaly_idxs[0]
    anoms = []
    noms = []
    anom_correct = []
    noms_correct = []
    for i in anomaly_idxs[0]:
        label = data.iloc[i]["label"]
        if label == "nacl": anom_correct.append(True)
        else: anom_correct.append(False)
        anoms.append(distances.mean(axis=1)[i])
    for i in nominal_idxs[0]:
        label = data.iloc[i]["label"]
        if label == "kcl": noms_correct.append(True)
        else: noms_correct.append(False)
        noms.append(distances.mean(axis=1)[i])
    print(100*"*")
    print("ANOMALY DETECTION COMPLETE")
    print(f"True Positive Rate: {round(100*sum(anom_correct)/len(anom_correct))}%")
    print(f"False Positive Rate: {round(100*sum(invert(anom_correct))/len(anom_correct))}%")
    print(f"True Negative Rate: {round(100*sum(noms_correct)/len(noms_correct))}%")
    print(f"False Negative Rate: {round(100*sum(invert(noms_correct))/len(noms_correct))}%")
    #==========================================================================
    #   Plot the results
    #==========================================================================
    _, axes = plt.subplots(1, 1, figsize=(3.25, 3.25))
   # axes.scatter(indexes, di)
    plt.savefig("images/elec_anomaly.png")

#==============================================================================
if __name__ == "__main__": elec_anomaly()