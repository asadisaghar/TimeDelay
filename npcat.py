import numpy as np
import sys

inputs = sys.argv[1:-1]
output = sys.argv[-1]

data = None
for input in inputs:
    varname = 'arr_0'
    if ':' in input:
        input, varname = input.split(":")
    data1 = np.load(input)
    if isinstance(data1, np.lib.npyio.NpzFile):
        data1 = data1[varname]
    if data is None:
        data = data1
    else:
        data = np.append(data, data1)

if output.endswith(".npz"):
    np.savez(output, data)
else:
    np.save(output, data)
