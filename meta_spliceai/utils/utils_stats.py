import numpy as np


def five_number_stats(values):
    values = np.array(values)
    min, max = np.min(values), np.max(values)
    quartiles = np.percentile(values, [25, 50, 75])
    return np.hstack((min, quartiles, max))