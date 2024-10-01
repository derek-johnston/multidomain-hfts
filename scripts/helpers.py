#==============================================================================
#   Dependencies
#==============================================================================
from numpy                      import where
from numpy.random               import normal
from pandas                     import read_csv
from sklearn.neighbors          import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection    import train_test_split
from sklearn.metrics            import confusion_matrix, ConfusionMatrixDisplay
#==============================================================================
def get_data(path, label="", noise=True):
    """Read-in and parse an HFTS dataset."""
    data = read_csv(path)
    data = data.drop("Time", axis=1)
    data = data.astype(complex)
    data = data.apply(abs)
    if noise:
        data += normal(0, 0.0025, data.shape)
    if label != "":
        data["label"] = label
    return data
#==============================================================================
def gen_model(dataset):
    """Generate a KNN model from the given dataset"""
    X = dataset.drop("label", axis=1).to_numpy()
    y = dataset["label"].to_numpy()
    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    print(f"Model Trained with score = {model.score(X_test, y_test)}")
    # Generate and return the confusion matrix
    labels = list(set(y))
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)
    return ConfusionMatrixDisplay(
        confusion_matrix=matrix, 
        display_labels=labels
    )
#==============================================================================
def plot_cm(cm, axes):
    """Generate a confusion matrix plot"""
    cm.plot(ax=axes, cmap="Greys", colorbar=False, xticks_rotation="vertical")
#==============================================================================
def detect_anomalies(dataset, real="kcl", fake="nacl", threshold=0.0019):
    """Use unsupervised learning to detect anomalous samples."""
    # Perform the nearest neighbor calculation
    X = dataset.drop("label", axis=1).to_numpy()
    y = dataset["label"].to_numpy()
    model = NearestNeighbors().fit(X)
    distances, _ = model.kneighbors(X)
    # Compute the confusion matrix
    anom_idxs   = where(distances.mean(axis=1) > threshold)
    noms_idxs   = where(distances.mean(axis=1) <= threshold)
    true_pos    = []
    false_pos   = []
    true_neg    = []
    false_neg   = []
    for i in anom_idxs[0]:
        dist = distances.mean(axis=1)[i]
        if dataset.iloc[i]["label"] == fake: true_pos.append((i, dist))
        else: false_pos.append((i, dist))
    for i in noms_idxs[0]:
        dist = distances.mean(axis=1)[i]
        if dataset.iloc[i]["label"] == real: true_neg.append((i, dist))
        else: false_neg.append((i, dist))
    TP = round(100*len(true_pos)/len(anom_idxs[0]))
    FP = round(100*len(false_pos)/len(anom_idxs[0]))
    TN = round(100*len(true_neg)/len(noms_idxs[0]))
    FN = round(100*len(false_neg)/len(noms_idxs[0]))
    print(100*"*")
    print("ANOMALY DETECTION")
    print(100*"*")
    print(f"True Positives = {TP}%")
    print(f"False Positives = {FP}%")
    print(f"True Negatives = {TN}%")
    print(f"False Negatives = {FN}%")
    print(100*"*")
    return {
        "n_tp": TP,
        "n_fp": FP,
        "n_tn": TN,
        "n_fn": FN,
        "t_pos": true_pos,
        "f_pos": false_pos,
        "t_neg": true_neg,
        "f_neg": false_neg
    }

#==============================================================================
