import sys
import numpy as np

input = sys.argv[1]
output = sys.argv[2]

data = np.load(input)['arr_0']
pair_ids = np.unique(data['pair_id'])

res = np.zeros(len(pair_ids), dtype=[('pair_id', 'f4'), ('est_dt_mean', 'f4'), ('est_dt_std', 'f4'), ('dt', '<f4'), ('tau', '<f4'), ('sig', '<f4'), ('m1', '<f4'), ('m2', '<f4')])

for j, pair_id in enumerate(pair_ids):
    print "%s: %s of %s" % (pair_id, j, len(pair_ids))

    pair_data = data[data['pair_id'] == pair_id]
    windows = np.unique(pair_data['window_id'])

    est_dts = np.zeros(len(windows))
    for i, window in enumerate(windows):
        print "Window %s" % window
        window_data = pair_data[pair_data['window_id']==window]
        est_dts[i] = window_data['offset'][np.argmax(window_data['correlation'])]

    res['pair_id'] = pair_id
    res['est_dt_mean'] = est_dts.mean()
    res['est_dt_std'] = est_dts.std()
    for name in ('pair_id', 'dt', 'tau', 'sig', 'm1', 'm2'):
        res[name] = pair_data[name][0]

np.savez(output, res)
