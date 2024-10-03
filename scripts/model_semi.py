#==============================================================================
#   Dependencies
#==============================================================================
from pickle                     import dump, load
from sklearn.model_selection    import train_test_split
from sklearn.neighbors          import KNeighborsClassifier
#==============================================================================
def model_semi(classes=["microcontroller", "timer"]):
    """Generate an array of classification models for semiconductors"""
    # Read-in the data from the pickles.
    p_filename = f"semi_d"
    for c in classes: 
        p_filename += f"_{c}"
    p_filename += ".pkl"
    with open(f"pickles/{p_filename}", "rb") as file:
        datasets = load(file)
    # Train the models
    models = []
    t_scores = []
    for sample in datasets:
        X = sample.drop("label", axis=1)
        y = sample["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        t_scores.append(score)
        print(f"Model trained with score = {score}")
        models.append(model)
    # Pickle the models
    p_filename = f"semi_m"
    for c in classes: 
        p_filename += f"_{c}"
    p_filename += ".pkl"
    with open(f"pickles/{p_filename}", "wb") as file:
        dump(models, file)
    p_filename = f"semi_t"
    for c in classes: 
        p_filename += f"_{c}"
    p_filename += ".pkl"
    with open(f"pickles/{p_filename}", "wb") as file:
        dump(t_scores, file)
#==============================================================================
if __name__ == "__main__":
    model_semi(classes=["cc8z", "d855"])
#==============================================================================