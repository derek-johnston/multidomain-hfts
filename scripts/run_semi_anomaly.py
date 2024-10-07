#==============================================================================
#   Dependencies
#==============================================================================
from semi_anomaly_getdata       import semi_anomaly_getdata
from semi_anomaly               import semi_anomaly
from helpers                    import dump_pickle_aresults
from plot_semi_anomaly          import plot_semi_anomaly
#==============================================================================
def run_semi_anomaly(t_mult=0.5):
    """Run the semi-anomaly detection pipeline"""
    classes = [
        ["microcontroller", "timer"],
        ["2210", "2230"],
        ["cc8z", "d855"],
        ["ckdf", "dont"],
        ["555cm", "555im"],
        ["78j9", "93y8"],
    ]
    #for c in classes:
    #    semi_anomaly_getdata(classes=c)
    results = []
    for c in classes:
        print(100*"*")
        print(f"Anomaly detection pipeline for classes: {c}")
        results.append(semi_anomaly(classes=c, t_mult=t_mult))
    return results
#==============================================================================
if __name__ == "__main__":
    results = []
    for t in [0.3, 0.6, 0.9]:
        results.append(run_semi_anomaly(t_mult=t))
    dump_pickle_aresults(results)
    plot_semi_anomaly()
#==============================================================================