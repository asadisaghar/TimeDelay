import numpy as np
from matplotlib import pyplot as plt
import sklearn.linear_model

x = np.load("TimeDelayData/fft_comparisons_with_truth.npz")['arr_0']

xcols = [name for name in x.dtype.names if name.startswith("fft_")]

X = np.zeros((len(x), len(xcols)))
for i, xcol in enumerate(xcols):
    X[:,i] = x[xcol]
 
m = sklearn.linear_model.LinearRegression()
m.fit(X, x['dt'])

ypred = m.predict(X)

print "MSE: %s" % (sum((x['dt'] - ypred)**2) / len(x),)

plt.plot(x['dt'], ypred, '.')
plt.show()
