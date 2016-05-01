import sys
import numpy as np

input = sys.argv[1]
output = sys.argv[2]

data = np.load(input)['arr_0']
pair_ids = np.unique(data['pair_id'])

res = np.zeros(len(pair_ids), dtype=[('pair_id', 'f4'), ('est_dt_mean', 'f4'), ('est_dt_median', 'f4'), ('est_dt_std', 'f4'), ('est_dt_wgtd_mean', 'f4'), ('dt', '<f4'), ('tau', '<f4'), ('sig', '<f4'), ('m1', '<f4'), ('m2', '<f4')])

for j, pair_id in enumerate(pair_ids):
    print "%s: %s of %s" % (pair_id, j, len(pair_ids))

    pair_data = data[data['pair_id'] == pair_id]
    windows = np.unique(pair_data['window_id'])
    window_wgts = np.zeros(len(windows))
    est_dts = np.zeros(len(windows))
    for i, window in enumerate(windows):
        print "Window %s" % window
        window_data = pair_data[pair_data['window_id']==window]
        est_dts[i] = window_data['offset'][np.argmax(window_data['correlation'])]
        window_wgts[i] = np.max(window_data['correlation'])/len(window_data)

    res['pair_id'][j] = pair_id
    res['est_dt_mean'][j] = est_dts.mean()
    res['est_dt_median'][j] = np.median(est_dts)
    res['est_dt_std'][j] = est_dts.std()
    res['est_dt_wgtd_mean'][j] = np.mean((est_dts * window_wgts) / np.sum(window_wgts))
    for name in ('pair_id', 'dt', 'tau', 'sig', 'm1', 'm2'):
        res[name][j] = pair_data[name][0]

np.savez(output, res)
