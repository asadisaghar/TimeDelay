import numpy as np; import matplotlib.pyplot as plt
import sklearn.linear_model
from numpy.lib.recfunctions import *

x = np.load("TimeDelayData/timeshift_correlate_normalized_detrend.measures.npz")['arr_0']
x = append_fields(x, 'ms', [], dtypes='<f4')
x['ms'] = x['est_dt_mean'] * x['est_dt_std']

dts = np.unique(x['dt'])

msmean = np.zeros(len(dts))
for i in xrange(0, len(dts)):
    msmean[i] = x[x['dt'] == dts[i]]['ms'].mean()

def findcluster(x):
    res = np.zeros(len(x))
    for i in xrange(0, len(x)):
        res[i] = (1 / (1 + x[i] - x)).sum()
    return x[np.argmax(res)]
clusters = np.zeros(len(dts))
for i in xrange(0, len(dts)):
    clusters[i] = findcluster(x[x['dt'] == dts[i]]['ms'])

def findcluster2(x):
    res = np.zeros(len(x))
    for i in xrange(0, len(x)):
        res[i] = (1 / (1 + (x[i] - x)**2)).sum()
    return x[np.argmax(res)]
clusters2 = np.zeros(len(dts))
for i in xrange(0, len(dts)):
    clusters2[i] = findcluster2(x[x['dt'] == dts[i]]['ms'])

    

plt.plot(x['dt'], x['ms'] / x['ms'].std(), 'g.')
#plt.plot(dts, msmean / msmean.std(), 'r.')
#plt.plot(dts, clusters / clusters.std(), 'r.')
plt.plot(dts, clusters2 / clusters2.std(), 'b.')
plt.show()


# X = np.zeros((len(clusters), 5))
# X[:,0] = clusters
# X[:,1] = clusters ** 2
# X[:,2] = clusters ** 3
# X[:,3] = clusters ** 4
# X[:,4] = clusters ** 5
# m = sklearn.linear_model.LinearRegression()
# m.fit(X, dts)

# dtspred = m.predict(X)

# plt.plot(clusters, dts, 'r.')
# plt.plot(clusters, dtspred, 'g.')
# plt.show()
