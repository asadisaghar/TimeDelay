import numpy as np
from timedelay.sampling_windows import *
from numpy.lib.recfunctions import *

data = np.load("TimeDelayData/pairs_with_truths.npz")['arr_0']
pair_ids = np.unique(data['full_pair_id'])

data = append_fields(data, 'window_id', [], dtypes='<f8')
data = data.filled()

window_id = 0
for pair_id in pair_ids:
    pair_query = data['full_pair_id'] == pair_id
    pair_data = data[pair_query]
    
    times = data[pair_query]['time']
    times.sort()
    
    time_windows = times[pick_sampling_windows(times)]
    
    for idx in xrange(0, len(time_windows)):
        window_start, window_end = time_windows[idx]
        data['window_id'][pair_query
                          & (data['time'] >= window_start)
                          & (data['time'] <= window_end)] = window_id
        window_id += 1

np.savez("TimeDelayData/pairs_with_truths_and_windows.npz", data)
