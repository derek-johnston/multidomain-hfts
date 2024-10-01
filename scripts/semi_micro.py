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
        if "microcontroller" in file:
            data = read_csv(f"data/semiconductor/{file}", header=None)
            print(f"Reading in {file}")
            for i in range(64):
                df = datasets[i]
                row = data.iloc[i]
                if "2230" in file: row["label"] = "2230"
                else: row["label"] = "2210"
                df = df._append(row, ignore_index=True)
                datasets[i] = df
    #==========================================================================
    #   Perform the classification
    #==========================================================================
    predictions = []
    for i in range(64):
        X = datasets[i].drop("label", axis=1).to_numpy()
        y = datasets[i]["label"].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        predict = model.predict(X_test)
        print(f"({i+1}) Predictions = {predict}")
        predictions.append(predict)
    print(predictions)
    print(y_test)
    counts = [{"2230":0, "2210":0} for _ in range(len(y_test))]
    print(counts)
    for i in range(len(y_test)):
        for j in range(64):
            #p = predictions[j][i]
            #print(f"Prediction = {p}")
            counts[i][predictions[j][i]] += 1

    """
    counts = []
    for j in range(len(y_test)):
        counts.append({"2230": 0, "2210": 0})
        for i in range(64):
            for p in predictions[j]:
                print(p)
                counts[j][p] = counts[j][p] + 1
    """
    
    print(100*"*")
    print("CLASSIFICATION RESULTS")
    count = 0
    for i in range(len(y_test)):
        print(100*"*")
        if counts[i]["2230"] > counts[i]["2210"]: pick = "2230"
        else: pick = "2210"
        if pick == y_test[i]: count += 1
        print(f"({i+1}) Label = {y_test[i]} | Pick = {pick} | 2230 = {counts[i]["2230"]} | 2210 = {counts[i]["2210"]}")
        print(100*"*")
    print(f"Result = {round(100*count/len(y_test))}%")
    print(100*"*")
    
#==============================================================================
if __name__ == "__main__": semi_device()