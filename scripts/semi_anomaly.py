#==============================================================================
#   Dependencies
#==============================================================================
from collections                import Counter
from helpers                    import load_pickle_adata
from sklearn.neighbors          import NearestNeighbors
from numpy                      import mean, std, where, max
#==============================================================================
def semi_anomaly(classes=["microcontroller", "timer"], t_mult=1.0):
    """Perform anomaly detection on HFTS semiconductor data"""
    # Load the datasets
    dataset = load_pickle_adata(classes=classes)
    N = len(dataset[0])
    # Perform anomaly detection
    counter_a = Counter()
    for i, data in enumerate(dataset):
        X = data.drop("label", axis=1).to_numpy()
        y = data["label"].to_numpy()
        y_idx = where(y == classes[1])[0]
        neighbors = NearestNeighbors().fit(X)
        distances, _ = neighbors.kneighbors(X)
        threshold = mean(distances) + (std(distances) * t_mult)
        anom_idx = where(distances > threshold)[0]
        counter_a += Counter(anom_idx.tolist())
    anomalies = []
    normal = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(N):
        score = round(100 * counter_a[i] / (2 * 64))
        if score >= 50: 
            anomalies.append(i)
            if y[i] == classes[1]: tp += 1
            elif y[i] == classes[0]: fp += 1
        else:
            normal.append(i)
            if y[i] == classes[0]: tn += 1
            elif y[i] == classes[1]: fn += 1

    print(100*"*")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(100*"*")
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "t_mult": t_mult,
        "classes": classes
    }
#==============================================================================
if __name__ == "__main__":
    semi_anomaly()
#==============================================================================