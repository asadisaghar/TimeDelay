import numpy as np; import matplotlib.pyplot as plt

x = np.load("TimeDelayData/timeshift_correlate_normalized_detrend.measures.npz")['arr_0']

plt.plot(x['dt'], x['est_dt_mean'] * x['est_dt_std'], '.'); plt.show()


dts = np.unique(x['dt'])
x2 = np.zeros(len(dts), dtype=[('dt', 'f4'), ('mean', 'f4'), ('std', 'f4')])
x2['dt'] = dts
for i in xrange(0, len(dts)):
    x2['std'][i] = x[x['dt'] == dts[i]]['est_dt_mean'].std()
    x2['mean'][i] = x[x['dt'] == dts[i]]['est_dt_mean'].mean()

plt.plot(x2['dt'], x2['mean'], 'r.')
plt.plot(x2['dt'], x2['std'], 'g.')
plt.show()
