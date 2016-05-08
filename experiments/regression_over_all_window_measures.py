import sys
import numpy as np
from matplotlib import pyplot as plt
import sklearn.linear_model
import numpy.lib.recfunctions
import cPickle as pickle
import timedelay.features

data = numpy.load("TimeDelayData/timeshift_correlate_normalized_detrend.measures.npz")['arr_0']
pair_ids = np.unique(data['pair_id'])
windows_per_pair = len(data) / len(pair_ids)

X = np.zeros((len(pair_ids), windows_per_pair))
y = np.zeros(len(pair_ids))
for idx, pair_id in enumerate(pair_ids):
    pair_data = data[data['pair_id'] == pair_id]
    pair_data = pair_data[np.argsort(-pair_data['est_weight'])]
    est_dts = pair_data['est_dt'][:windows_per_pair]
    X[idx,:len(est_dts)] = est_dts 
    y[idx] = pair_data['dt'][0]

trainlen = np.ceil(len(X) * 3. / 4.)
Xtrain = X[:trainlen]
ytrain = y[:trainlen]
Xtest = X[trainlen:]
ytest = y[trainlen:]

m = sklearn.linear_model.LinearRegression()
m.fit(Xtrain, ytrain)

ypred = m.predict(Xtest)

plt.plot(ytest, ypred, '.b')
plt.plot(ytest, ytest, '-r')
plt.ylim(-150,150)
plt.xlim(-150,150)
plt.show()
