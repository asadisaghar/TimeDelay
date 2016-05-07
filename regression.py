import sys
import numpy as np
from matplotlib import pyplot as plt
import sklearn.linear_model
import numpy.lib.recfunctions

if len(sys.argv) < 2:
    print """Usage: regression.py input_field1,...,input_fieldN output_field file1 ... fileM output"""
    sys.exit(-1)

input_fields = sys.argv[1].split(",")
output_field = sys.argv[2]
inputs = sys.argv[3:-1]
output = sys.argv[-1]

srccols = []
y = None
groundtruth = None
for i, input in enumerate(inputs):
    data = np.load(input)['arr_0']
    if y is None:
        y = data[output_field]
    if groundtruth is None:
        groundtruth = data[['pair_id', 'dt', 'tau', 'sig', 'm1', 'm2']]
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


randidx = np.arange(0, len(X), 1)
np.random.shuffle(randidx)

X = X[randidx]
y = y[randidx]
groundtruth = groundtruth[randidx]

num_rows = len(X)
train_len = int(num_rows * 3. / 4.)
Xtrain = X[:train_len,:]
ytrain = y[:train_len]
Xtest = X[train_len:,:]
ytest = y[train_len:]
groundtruthtest = groundtruth[train_len:]

m = sklearn.linear_model.LinearRegression()
m.fit(Xtrain, ytrain)

with open(output + ".model.pkl", "wb") as f:
    pickle.dump(m, f)

ypred = m.predict(Xtest)

print "MSE: %s" % (sum((ytest - ypred)**2) / len(ypred),)


res = np.zeros(len(ypred), dtype=[('pair_id', 'f4'), ('est_dt', 'f4'), ('dt', '<f4'), ('tau', '<f4'), ('sig', '<f4'), ('m1', '<f4'), ('m2', '<f4')])

for name in ('pair_id', 'dt', 'tau', 'sig', 'm1', 'm2'):
    res[name] = groundtruthtest[name]
res['est_dt'] = ypred

np.savez(output + ".predictions.npz", x = res)

