import sys
import numpy as np

input = sys.argv[1]
output = sys.argv[2]

data = np.load(input)['arr_0']
pair_ids = np.unique(data['pair_id'])

res = None
for j, pair_id in enumerate(pair_ids):
    print "%s: %s of %s" % (pair_id, j, len(pair_ids))

    pair_data = data[data['pair_id'] == pair_id]
    windows = np.unique(pair_data['window_id'])

    pair_res = np.zeros(len(windows), dtype=[('pair_id', 'f4'), ('window_id', 'f4'), ('est_dt', 'f4'), ('est_weight', 'f4'), ('dt', '<f4'), ('tau', '<f4'), ('sig', '<f4'), ('m1', '<f4'), ('m2', '<f4')])
    for i, window in enumerate(windows):
        print "Window %s" % window
        window_data = pair_data[pair_data['window_id']==window]

        pair_res['est_dt'][i] = window_data['offset'][np.argmax(window_data['correlation'])]
        pair_res['est_weight'][i] = np.max(window_data['correlation'])/len(window_data)
        pair_res['window_id'][i] = window

    pair_res['pair_id'] = pair_id
    for name in ('pair_id', 'dt', 'tau', 'sig', 'm1', 'm2'):
        pair_res[name] = pair_data[name][0]

    if res is None:
        res = pair_res
    else:
        res = np.append(res, pair_res)

np.savez(output, res)
