#==============================================================================
#   Dependencies
#==============================================================================
import  matplotlib.pyplot   as plt
from    pandas              import concat
from    helpers             import get_data, detect_anomalies, plot_anomaly
#==============================================================================
def mp_anomaly():
    """Detect anomalous samples in microparticle data"""
    #==========================================================================
    #   Read-in the data.
    #==========================================================================
    root        = "data/microparticle/microparticle_amidine_"
    amidine1    = get_data(f"{root}01.csv", label="amidine")
    amidine2    = get_data(f"{root}02.csv", label="amidine")
    amidine3    = get_data(f"{root}03.csv", label="amidine")
    root        = "data/microparticle/microparticle_carbox_"
    carbox1     = get_data(f"{root}01.csv", label="carbox")
    carbox2     = get_data(f"{root}02.csv", label="carbox")
    carbox3     = get_data(f"{root}03.csv", label="carbox")
    amidine     = concat([amidine1, amidine2, amidine3])
    carbox      = concat([carbox1, carbox2, carbox3])
    #==========================================================================
    #   Compile the data
    #==========================================================================
    data = concat([
        carbox.sample(frac=0.90),
        amidine.sample(frac=0.10)
    ]).sample(frac=1)
    print(data.shape)
    print(amidine.sample(frac=0.9).shape)
    print(carbox.sample(frac=0.1).shape)
    #==========================================================================
    #   Detect and plot anomalies
    #==========================================================================
    threshold = 0.032
    best_threshold = 0.000
    max_avg = 0
    while threshold < 0.042:
        print(f"Threshold = {threshold}")
        results = detect_anomalies(data, real="carbox", fake="amidine", threshold=threshold)
        avg = (results["n_tp"] + results["n_tn"]) / 2
        print(f"{avg}")
        if avg > max_avg and avg < 95: 
            max_avg = (results["n_tp"] + results["n_tn"]) / 2
            best_threshold = threshold
        threshold = round(threshold + 0.0001, 4)
    print(100*"*")
    print(f"Best threshold = {best_threshold}")
    print(100*"*")
    plot_anomaly(
        detect_anomalies(data, real="carbox", fake="amidine", threshold=best_threshold),
        threshold=best_threshold,
        filename="mp_anomaly"
    )
#==============================================================================
if __name__ == "__main__": mp_anomaly()