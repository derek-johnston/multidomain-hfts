#==============================================================================
#   Dependencies
#==============================================================================
import  matplotlib.pyplot       as plt
from    os                      import listdir
from    pandas                  import concat, read_csv, DataFrame
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.model_selection    import train_test_split
#==============================================================================
def semi_device():
    """Classify devices based on device type"""
    #==========================================================================
    #   Prepare the data structure to hold the data.
    #==========================================================================
    datasets = [DataFrame() for _ in range(64)]
    #==========================================================================
    #   Read-in the data.
    #==========================================================================
    for file in listdir("data/semiconductor"):
        data = read_csv(f"data/semiconductor/{file}", header=None)
        for i in range(64):
            df = datasets[i]
            row = data.iloc[i]
            if "2210" in file: row["label"] = "2210"
            else: row["label"] = "2230"
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
        score = model.score(X_test, y_test)
        print(f"({i+1}) Model Trained with score = {score}")
        scores.append(score)
    print(scores)
    
#==============================================================================
if __name__ == "__main__": semi_device()