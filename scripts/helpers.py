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
def detect_anomalies(dataset, threshold=0.0019):
    """Use unsupervised learning to detect anomalous samples."""
    # Perform the nearest neighbor calculation
    X = dataset.drop("label", axis=1).to_numpy()
    y = dataset["label"].to_numpy()
    model = NearestNeighbors().fit(X)
    distances, _ = model.kneighbors(X)
    # Compute the confusion matrix
    anom_idxs = where(distances.mean(axis=1))
#==============================================================================
