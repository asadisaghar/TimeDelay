import sys
import numpy as np
from matplotlib import pyplot as plt
import sklearn.linear_model
import numpy.lib.recfunctions
import cPickle as pickle
import timedelay.features

if len(sys.argv) < 2:
    print """Usage: predict.py input_field1,...,input_fieldN output_field file1 ... fileM model"""
    sys.exit(-1)

input_fields = sys.argv[1].split(",")
degree = int(sys.argv[2])
output_field = sys.argv[3]
input_files = sys.argv[4:-1]
model_file = sys.argv[-1]

with open(model_file, "rd") as f:
    m = pickle.load(f)

X, y = timedelay.features.load_features(input_fields, output_field, input_files, degree)

ypred = m.predict(X)

import matplotlib.pyplot as plt

plt.plot(y, ypred, '.g')
plt.plot(y, y, '-r')
plt.ylim(-150, 150)
plt.xlim(-150, 150)
plt.show()
