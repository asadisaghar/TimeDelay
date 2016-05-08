import numpy as np
from matplotlib import pyplot as plt
import sklearn.linear_model

x = np.load("TimeDelayData/fft_comparisons_with_truth.npz")['arr_0']
np.random.shuffle(x)

xcols = [name for name in x.dtype.names if name.startswith("fft_")]

X = np.zeros((len(x), len(xcols)))
for i, xcol in enumerate(xcols):
    X[:,i] = x[xcol]
y = x['dt']

num_rows = len(X)
train_len = int(num_rows * 3. / 4.)
Xtrain = X[:train_len,:]
ytrain = y[:train_len]
Xtest = X[train_len:,:]
ytest = y[train_len:]


m = sklearn.linear_model.LinearRegression()
m.fit(Xtrain, ytrain)

ypred = m.predict(Xtest)

print "MSE: %s" % (sum((ytest - ypred)**2) / len(ypred),)

plt.plot(ytest, ypred, '.')
plt.show()
