import sys
import os
sys.path[0:0] = [os.path.dirname(os.path.dirname(__file__))]
import numpy as np
from matplotlib import pyplot as plt
import sklearn.linear_model
import numpy.lib.recfunctions
import cPickle as pickle
import timedelay.features

if len(sys.argv) < 2:
    print """Usage: predict.py input_field1,...,input_fieldN output_field file1 ... fileM model output"""
    sys.exit(-1)

input_fields = sys.argv[1].split(",")
degree = int(sys.argv[2])
output_field = sys.argv[3]
input_files = sys.argv[4:-2]
model_file = sys.argv[-2]
output_file = sys.argv[-1]

with open(model_file, "rd") as f:
    m = pickle.load(f)

X, y = timedelay.features.load_features(input_fields, output_field, input_files, degree)
pair_ids = np.load(input_files[0])['arr_0']['pair_id']

trainlen = np.ceil(len(X) * 3 / 4)
X = X[trainlen:]
y = y[trainlen:]
pair_ids = pair_ids[trainlen:]

ypred = m.predict(X)

res = np.zeros(len(y), dtype=[('pair_id', 'f4'), ('dt', 'f4'), ('est', 'f4'), ('est_err', 'f4')])
res['dt'] = y
res['est'] = ypred
res['est_err'] = 0 # FIXME....

np.savez(output_file, res)

mse = ((y - ypred)**2).sum()/len(y)

print "MSE: %s" % mse

import matplotlib.pyplot as plt

plt.plot(y, ypred, '.g')
plt.plot(y, y, '-r')
plt.ylim(-150, 150)
plt.xlim(-150, 150)
plt.show()
