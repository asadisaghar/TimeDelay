import sys
import numpy as np
from matplotlib import pyplot as plt
import sklearn.linear_model
import numpy.lib.recfunctions

def load_features(input_fields, output_field, input_files, degrees = 2):
    print "XXXXX", input_fields, output_field, input_files, degrees
    srccols = []
    y = None
    for i, input in enumerate(input_files):
        data = np.load(input)['arr_0']
        if y is None:
            y = data[output_field]
        for input_field in input_fields:
            srccols.append(data[input_field])

    signs = [np.sign(col) for col in srccols]
    srccols += signs
    cols = list(srccols)
    for deg in xrange(1, degrees):
        rescols = []
        for srccol in srccols:
            for col in cols:
                rescols.append(col * srccol)
        cols += rescols

    X = np.zeros((len(data), len(cols)))
    for i, col in enumerate(cols):
        X[:,i] = col

    return X, y
