#==============================================================================
#   Dependencies
#==============================================================================
from numpy.random               import normal
from os                         import listdir
from pandas                     import concat, read_csv, DataFrame
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.model_selection    import train_test_split
#==============================================================================
def semi_classify(class0="microcontroller", class1="timer", noise=0.0025):
    """Classify HFTS semiconductor datasets"""
    #==========================================================================
    #   Read-in the data
    #==========================================================================
    data = [DataFrame() for _ in range(64)]
    data0 = DataFrame()
    data1 = DataFrame()
    for filename in listdir("data/semiconductor"):
        if class0 in filename:
            print(f"Reading in {filename}")
            data0 = read_csv(f"data/semiconductor/{filename}", header=None)
            data0 += normal(0, noise, data0.shape)
            data0["label"] = class0
            for i in range(64):
                data[i] = data[i]._append(data0.iloc[i, :], ignore_index=True)
        if class1 in filename:
            print(f"Reading in {filename}")
            data1 = read_csv(f"data/semiconductor/{filename}", header=None)
            data1 += normal(0, noise, data1.shape)
            data1["label"] = class1
            for i in range(64):
                data[i] = data[i]._append(data1.iloc[i, :], ignore_index=True)
    #==========================================================================
    #   Randomize the datasets
    #==========================================================================
    for i in range(64): data[i] = data[i].sample(frac=1)
    #==========================================================================
    #   Train the models
    #==========================================================================
    models = []
    for sample in data:
        X = sample.drop("label", axis=1)
        y = sample["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        print(f"Model {i} trained with score = {model.score(X_test, y_test)}")
        models.append(model)
    #==========================================================================
    #   Test the models
    #==========================================================================
    t_data = read_csv("data/semiconductor/microcontroller_2210_cc8z_1.csv", header=None)
    t_data += normal(0, noise, t_data.shape)
    t_data["label"] = "cc8z"
    counts = {class0: 0, class1: 0}
    for idx, model in enumerate(models):
        row = t_data.iloc[idx, :]
        row = row.to_frame().T
        X = row.drop("label", axis=1)
        y = row["label"]
        y_pred = model.predict(X)
        print(f"Model {idx+1} predicted {y_pred[0]}")
        counts[y_pred[0]] += 1
    print(f"Sample = {class0}: {counts}")
    #==========================================================================
    #   Generate the manuscript figures
    #==========================================================================
#==============================================================================
if __name__ == "__main__": semi_classify(class0="cc8z", class1="d855")
