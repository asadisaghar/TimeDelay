import sys
import numpy as np
from matplotlib import pyplot as plt
import sklearn.linear_model
import numpy.lib.recfunctions

if len(sys.argv) < 2:
    print """Usage: regression.py input_field1,...,input_fieldN output_field file1 ... fileM"""
    sys.exit(-1)

input_fields = sys.argv[1].split(",")
output_field = sys.argv[2]
inputs = sys.argv[3:]

srccols = []
y = None
for i, input in enumerate(inputs):
    data = np.load(input)['arr_0']
    if y is None:
        y = data[output_field]
    for input_field in input_fields:
        srccols.append(data[input_field])

degrees = 4
cols = list(srccols)
for deg in xrange(1, degrees):
    rescols = []
    for srccol in srccols:
        for col in cols:
            rescols.append(col * srccol)
    cols = rescols

X = np.zeros((len(data), len(cols)))
for i, col in enumerate(cols):
    X[:,i] = col

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
