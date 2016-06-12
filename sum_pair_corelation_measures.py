import sys
import numpy as np

input = sys.argv[1]
output = sys.argv[2]

data = np.load(input)['arr_0']
pair_ids = np.unique(data['pair_id'])

res = np.zeros(len(pair_ids), dtype=[('pair_id', 'f4'), ('est_dt_mean', 'f4'), ('est_dt_median', 'f4'), ('est_dt_std', 'f4'), ('est_dt_wgtd_mean', 'f4'), ('dt', '<f4'), ('tau', '<f4'), ('sig', '<f4'), ('m1', '<f4'), ('m2', '<f4')])

for j, pair_id in enumerate(pair_ids):
    print "%s: %s of %s" % (pair_id, j, len(pair_ids))

    # Remove windows that give no meaningful correlation
    pair_data = data[data['pair_id'] == pair_id]
    filtered_pair_data = pair_data[pair_data['est_dt'] != 0]
    if len(filtered_pair_data) > 0:
        pair_data = filtered_pair_data

    res['pair_id'][j] = pair_id
    res['est_dt_mean'][j] = pair_data['est_dt'].mean()
    res['est_dt_median'][j] = np.median(pair_data['est_dt'])
    res['est_dt_std'][j] = pair_data['est_dt'].std()
    res['est_dt_wgtd_mean'][j] = np.mean((pair_data['est_dt'] * pair_data['est_weight']) / np.sum(pair_data['est_weight']))
    for name in ('pair_id', 'dt', 'tau', 'sig', 'm1', 'm2'):
        res[name][j] = pair_data[name][0]

np.savez(output, res)
