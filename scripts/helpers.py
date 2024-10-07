#==============================================================================
#   Dependencies
#==============================================================================
import matplotlib.pyplot        as plt
from matplotlib.gridspec        import GridSpec
from os                         import listdir
from numpy                      import where
from numpy.random               import normal
from pandas                     import read_csv, DataFrame
from pickle                     import dump, load      
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
    if len(anom_idxs[0]) > 0:
        TP = round(100*len(true_pos)/len(anom_idxs[0]))
        FP = round(100*len(false_pos)/len(anom_idxs[0]))
    else:
        TP = 0
        FP = 0
    if len(noms_idxs[0]) > 0:
        TN = round(100*len(true_neg)/len(noms_idxs[0]))
        FN = round(100*len(false_neg)/len(noms_idxs[0]))
    else:
        TN = 0
        FN = 0
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
        "f_neg": false_neg,
        "a_idx": anom_idxs,
        "n_idx": noms_idxs
    }
#==============================================================================
def plot_anomaly(results, threshold, filename):
    fig = plt.figure(figsize=(10, 3.25))
    plt.rcParams.update({"font.size": 9})
    gs = GridSpec(1, 6, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0:4])
    ax1 = fig.add_subplot(gs[0, 4:6])
    # Confusion Matrix
    cm = [[results["n_tp"], results["n_fp"]], [results["n_fn"], results["n_tn"]]]
    ax1.imshow(cm, cmap="Greys")
    for i in range(2):
        for j in range(2):
            if cm[i][j] > 50: color="white"
            else: color="black"
            ax1.annotate(f"{str(cm[i][j])}%", xy=(j, i), ha="center", va="center", color=color, fontsize=16)
    ax1.set_title("(b)")
    ax1.set_yticks([0, 1], ["Anomaly", "Normal"])
    ax1.set_xticks([0, 1], ["Anomaly", "Normal"])
    ax1.set_ylabel("")
    ax1.set_xlabel("")
    # Scatter Plot
    color_true = "black"
    color_false = "dimgray"
    for r, i in zip(results["t_pos"], range(len(results["t_pos"]))):
        if i == 0: ax0.scatter(r[0], r[1], color=color_true, marker="*", label="True Positive")
        else: ax0.scatter(r[0], r[1], color=color_true, marker="*")
    for r, i in zip(results["t_neg"], range(len(results["t_neg"]))):
        if i == 0: ax0.scatter(r[0], r[1], color=color_false, marker="*", label="True Negative")
        else: ax0.scatter(r[0], r[1], color=color_false, marker="*")
    for r, i in zip(results["f_pos"], range(len(results["f_pos"]))):
        if i == 0: ax0.scatter(r[0], r[1], color=color_true, marker="x", label="False Positive")
        else: ax0.scatter(r[0], r[1], color=color_true, marker="x")
    for r, i in zip(results["f_neg"], range(len(results["f_neg"]))):
        if i == 0: ax0.scatter(r[0], r[1], color=color_false, marker="x", label="False Negative")
        else: ax0.scatter(r[0], r[1], color=color_false, marker="x")
    ax0.axhline(y=threshold, color="black", linestyle="--")
    ax0.legend(frameon=False, loc="upper left", ncol=4)
    ax0.set_ylabel("Average Distance")
    ax0.set_xlabel("Sample Index")
    ax0.set_xlim([0, 1000])
    ax0.set_ylim([0.0008, 0.0042])
    ax0.set_title("(a)")
    plt.tight_layout()
    plt.savefig(f"images/{filename}.svg")
#==============================================================================
def semi_classify(root, class0, class1):
    """Classify semiconductor devices"""
    #==========================================================================
    #   Prepare the data structure to hold the data.
    #==========================================================================
    datasets = [DataFrame() for _ in range(64)]
    
    #==========================================================================
    #   Read-in the data.
    #==========================================================================
    for file in listdir("data/semiconductor"):
        if root in file:
            data = read_csv(f"data/semiconductor/{file}", header=None)
            print(f"Reading in {file}")
            for i in range(64):
                df = datasets[i]
                row = data.iloc[i]
                if class0 in file: row["label"] = class0
                else: row["label"] = class1
                df = df._append(row, ignore_index=True)
                datasets[i] = df
    #==========================================================================
    #   Perform the classification
    #==========================================================================
    scores = []
    for i in range(64):
        X = datasets[i].drop("label", axis=1).to_numpy()
        y = datasets[i]["label"].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        score = round(model.score(X_test, y_test), 2)
        #print(f"({i+1}) Model Trained with score = {score}")
        scores.append(score)
    return scores
#==============================================================================
def load_pickle_data(root="semi", classes=["microcontroller", "timer"]):
    """Read-in the data from the pickles."""
    p_filename = f"{root}_d"
    for c in classes: 
        p_filename += f"_{c}"
    p_filename += ".pkl"
    with open(f"pickles/{p_filename}", "rb") as file:
        datasets = load(file)
    return datasets
#==============================================================================
def load_pickle_model(root="semi", classes=["microcontroller", "timer"]):
    """Read-in the models from the pickles."""
    p_filename = f"{root}_m"
    for c in classes: 
        p_filename += f"_{c}"
    p_filename += ".pkl"
    with open(f"pickles/{p_filename}", "rb") as file:
        models = load(file)
    return models
#==============================================================================
def dump_pickle_results(results, root="semi", classes=["microcontroller", "timer"]):
    p_filename = f"semi_r"
    for c in classes: 
        p_filename += f"_{c}"
    p_filename += ".pkl"
    with open(f"pickles/{p_filename}", "wb") as file:
        dump(results, file)
#==============================================================================
def load_pickle_results(root="semi", classes=["microcontroller", "timer"]):
    """Read-in the results from the pickles."""
    p_filename = f"{root}_r"
    for c in classes: 
        p_filename += f"_{c}"
    p_filename += ".pkl"
    with open(f"pickles/{p_filename}", "rb") as file:
        results = load(file)
    return results
#==============================================================================
def load_pickle_tests(root="semi", classes=["microcontroller", "timer"]):
    """Read-in the test scores from the pickles."""
    p_filename = f"{root}_t"
    for c in classes: 
        p_filename += f"_{c}"
    p_filename += ".pkl"
    with open(f"pickles/{p_filename}", "rb") as file:
        tests = load(file)
    return tests
#==============================================================================
def dump_pickle_adata(data, root="semi", classes=["microcontroller", "timer"]):
    p_filename = f"semi_ad"
    for c in classes: 
        p_filename += f"_{c}"
    p_filename += ".pkl"
    with open(f"pickles/{p_filename}", "wb") as file:
        dump(data, file)
#==============================================================================
def load_pickle_adata(root="semi", classes=["microcontroller", "timer"]):
    """Read-in the test scores from the pickles."""
    p_filename = f"{root}_ad"
    for c in classes: 
        p_filename += f"_{c}"
    p_filename += ".pkl"
    with open(f"pickles/{p_filename}", "rb") as file:
        data = load(file)
    return data
#==============================================================================
def dump_pickle_aresults(data, root="semi", classes=["microcontroller", "timer"]):
    p_filename = f"semi_ar"
    for c in classes: 
        p_filename += f"_{c}"
    p_filename += ".pkl"
    with open(f"pickles/{p_filename}", "wb") as file:
        dump(data, file)
#==============================================================================
def load_pickle_aresults(root="semi", classes=["microcontroller", "timer"]):
    """Read-in the test scores from the pickles."""
    p_filename = f"{root}_ar"
    for c in classes: 
        p_filename += f"_{c}"
    p_filename += ".pkl"
    with open(f"pickles/{p_filename}", "rb") as file:
        data = load(file)
    return data