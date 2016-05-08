import numpy; import matplotlib.pyplot as plt

x = numpy.load("TimeDelayData/timeshift_correlate_normalized_detrend.measures.pairsums.clusters.npz")['arr_0']

def norm(a):
    return (a - a.mean()) / a.std()

plt.plot(x['dt'], norm(x['est_dt_wgtd_mean'] + x['est_dt_std'] * numpy.sign(x['est_dt_median'])), 'r.')
plt.plot(x['dt'], norm(x['est_dt_median'] + x['est_dt_std'] * numpy.sign(x['est_dt_median'])), 'g.')
plt.show()

# x = numpy.load("TimeDelayData/timeshift_correlate_normalized.measures.pairsums.clusters.npz")['arr_0']
# plt.plot(x['dt'], x['est_dt_median'] + x['est_dt_std'] * numpy.sign(x['est_dt_median']), '.')
# plt.show()

# x = numpy.load("TimeDelayData/timeshift_mse_normalized_detrend.measures.pairsums.clusters.npz")['arr_0']
# plt.plot(x['dt'], x['est_dt_median'] + x['est_dt_std'] * numpy.sign(x['est_dt_median']), '.')
# plt.show()

# x = numpy.load("TimeDelayData/timeshift_mse_normalized.measures.pairsums.clusters.npz")['arr_0']
# plt.plot(x['dt'], x['est_dt_median'] + x['est_dt_std'] * numpy.sign(x['est_dt_median']), '.')
# plt.show()
