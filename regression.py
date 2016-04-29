import sys
import numpy as np
from matplotlib import pyplot as plt
import sklearn.linear_model
import numpy.lib.recfunctions

if len(sys.argv < 2):
    print """Usage: regression.py input_field1,...,input_fieldN output_field file1 ... fileM"""
    sys.exit(-1)

input_fields = sys.argv[1].split(",")
output_field = sys.argv[2]
inputs = sys.argv[3:]

X = None
y = None
for i, input in enumerate(inputs):
    data = np.load(input)['arr_0']
    if X is None:
        X = np.zeros((len(data), len(input_fields)*len(inputs)))
        y = np.zeros(len(data))
        y[:] = data[output_field]
    for j, input_field in enumerate(input_fields):
        X[:,i * len(input_fields) + j] = data[input_field]

np.random.shuffle(X)

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
