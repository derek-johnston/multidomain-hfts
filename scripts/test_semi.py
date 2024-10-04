#==============================================================================
#   Dependencies
#==============================================================================
from pickle     import dump, load
from helpers    import load_pickle_data, load_pickle_model, dump_pickle_results
#==============================================================================
def test_semi(classes=["microcontroller", "timer"]):
    """Test the HFTS semiconductor classification models"""
    # Read-in the data from the pickles.
    datasets = load_pickle_data(root="semi", classes=classes)
    #return    # Read-in the models from the pickles
    models = load_pickle_model(root="semi", classes=classes)
    # Test the models
    scores = [0 for _ in range(64)]
    predictions = [[] for _ in range(64)]
    labels = ["" for _ in range(len(datasets[0]))]
    for i, (model, data) in enumerate(zip(models, datasets)):
        for j, row in data.iterrows():
            row = row.to_frame().T
            X = row.drop("label", axis=1)
            y = row["label"].iloc[0]
            labels[j] = y
            prediction = model.predict(X)
            predictions[i].append(prediction[0])
            score = 1 if prediction[0] == y else 0
            scores[i] += score
            print(f"Model {i+1} tested with score = {score}")
    
    # Pickle the results
    results = {
        "scores"        : scores,
        "predictions"   : predictions,
        "labels"        : labels
    }
    dump_pickle_results(results, root="semi", classes=classes)
#==============================================================================
if __name__ == "__main__":
    test_semi()
#==============================================================================