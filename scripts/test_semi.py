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
    scores = [[] for _ in range(64)]
    predictions = [[] for _ in range(64)]
    labels = [[] for _ in range(64)]
    for i, (model, data) in enumerate(zip(models, datasets)):
        print(f"Testing model {i+1}...")
        for j, row in data.iterrows():
            row = row.to_frame().T
            X = row.drop("label", axis=1)
            y = row["label"].iloc[0]
            labels[i].append(y)
            prediction = model.predict(X)
            predictions[i].append(prediction[0])
            score = 1 if prediction[0] == y else 0
            scores[i].append(score)
            #print(f"Model {i+1} tested with score = {score}")
    # Calculate the means and standard deviations   
    means = []
    sdevs = []
    for score in scores:
        means.append(round(sum(score) / len(score), 3))
        sdevs.append(round((sum([(s - means[-1])**2 for s in score]) / len(score))**0.5, 3))

    # Pickle the results
    results = {
        #"norm_scores"   : norm_scores,
        "scores"        : scores,
        "means"         : means,
        "sdevs"         : sdevs,
        "labels"        : labels,
        "predictions"   : predictions,
    }
    dump_pickle_results(results, root="semi", classes=classes)
#==============================================================================
if __name__ == "__main__":
    test_semi()
#==============================================================================