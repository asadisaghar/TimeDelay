import numpy as np
from matplotlib import pyplot as plt

reg = np.load("TimeDelayData/timeshift_correlate_normalized_detrend.measures.pairsums.regression-ests.npz")['arr_0']
heur = np.load("TimeDelayData/timeshift_correlate_normalized_detrend.measures.pairsums.ests.npz")['arr_0']

a = plt.hist(reg['est'] - reg['dt'], bins=50, histtype='step', normed=True, color='r', label='Regression')
a = plt.hist(heur['est'] - heur['dt'], bins=50, histtype='step', normed=True, color='b', label='Heuristic')
plt.legend()
plt.show()
