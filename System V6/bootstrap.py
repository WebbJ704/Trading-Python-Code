import numpy as np
import pandas as pd

def bootstrap(trades,n_simulations=1000):
    results = [] 
    for i in range(1, n_simulations):
        sample = trades['Return'].sample(len(trades), replace = True, random_state = i)
        sample_mean = sample.mean()
        sample_std = sample.std()
        sharp = sample_mean/sample_std
        results.append({"c0_means":sample_mean,
                        "c0_dev":sample_std,
                        "sharp":sharp})
    simulations = []
    for i in range(1, n_simulations):
        sample = trades['Return'].sample(len(trades), replace = True, random_state = i)
        cumulative_return = np.prod(1 + sample)
        simulations.append(cumulative_return)

    results = pd.DataFrame(results)
    simulations = pd.DataFrame(simulations)
    return results, simulations