import sys
import numpy as np

x = np.load(sys.argv[1])['arr_0']

res = np.zeros(len(x), dtype=[('pair_id', 'f4'), ('dt', 'f4'), ('est', 'f4'), ('est_err', 'f4')])
res['pair_id'] = x['pair_id']
res['dt'] = x['dt']
res['est'] = x['est_dt_median'] + x['est_dt_std'] * np.sign(x['est_dt_median'])
res['est_err'] = x['est_dt_std']

np.savez(sys.argv[2], res)
