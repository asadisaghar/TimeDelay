import sys
import numpy as np
from matplotlib import pyplot as plt
import sklearn.linear_model
import numpy.lib.recfunctions

inputs = sys.argv[1:]

data = None
for i, input in enumerate(inputs):
    input_data = np.load(input)['arr_0']
    if data is None:
        data = input_data
    else:
        data = append_fields(data, 'est_dt_mean_%s' % i, [], dtypes='<f4')
        data = append_fields(data, 'est_dt_std_%s' % i, [], dtypes='<f4')
        data['est_dt_mean_%s' % i] = input_data['est_dt_mean']
        data['est_dt_std_%s' % i] = input_data['est_dt_std']

np.random.shuffle(data)

datacols = [name for name in data.dtype.names if name.startswith("est_dt_")]

X = np.zeros((len(data), len(datacols)))
for i, datacol in enumerate(datacols):
    X[:,i] = data[datacol]
y = data['dt']

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
