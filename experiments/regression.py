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
    print """Usage: regression.py input_field1,...,input_fieldN degree output_field file1 ... fileM model"""
    sys.exit(-1)

input_fields = sys.argv[1].split(",")
degree = int(sys.argv[2])
output_field = sys.argv[3]
input_files = sys.argv[4:-1]
model_file = sys.argv[-1]

X, y = timedelay.features.load_features(input_fields, output_field, input_files, degree)

trainlen = np.ceil(len(X) * 3 / 4)
X = X[:trainlen]
y = y[:trainlen]

m = sklearn.linear_model.LinearRegression()
m.fit(X, y)

with open(model_file, "wb") as f:
    pickle.dump(m, f)
