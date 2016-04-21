import numpy as np
import matplotlib.pyplot as plt
from astroML.time_series import search_frequencies, lomb_scargle, MultiTermFit
import cPickle as pickle

data = np.load("TimeDelayData/gp_resampled_of_windows_with_truth.npz")['arr_0']
windows = np.unique(data['window_id'])[:1]

def compute_best_frequencies(windows, image='A', n_eval=10000, n_retry=5, generalized=True):
    results = {}
    for window in windows:
        t = data['t_eval'][data['window_id']==window]
        y = data['sig_eval%s'%(image)][data['window_id']==window]
        y = y - np.mean(y)
        dy = data['sig_err%s'%(image)][data['window_id']==window]
        dy = dy - np.mean(dy)
        print " - computing power for window %s (%s points)" % (window, len(t))
        kwargs = dict(generalized=generalized)
        omega, power = search_frequencies(t, y, dy, 
                                          n_eval=n_eval,
                                          n_retry=n_retry,
                                          LS_kwargs=kwargs)
        results[window] = [omega, power]

    return results

resultsA = compute_best_frequencies(windows, image='A')
with open("power_spectrum_frequencies_A.pkl", "wb") as f:
    pickle.dump(resultsA, f)

resultsB = compute_best_frequencies(windows, image='B')
with open("power_spectrum_frequencies_B.pkl", "wb") as f:
    pickle.dump(resultsB, f)

for i, window in enumerate(windows):
    plt.figure()
    ## A ##
    # get the data and best-fit angular frequency
    tA = data['t_eval'][data['window_id']==window]
    yA = data['sig_evalA'][data['window_id']==window]
    yA = (yA - np.mean(yA)) / np.std(yA)
    dyA = data['sig_errA'][data['window_id']==window]
    dyA = (dyA - np.mean(dyA)) / np.std(dyA)
    omegaA, powerA = resultsA[window]
    omega_bestA = omegaA[np.argmax(powerA)]
    print " - omega_0A = %.10g" % omega_bestA

    # do a fit to the first 4 Fourier components
    mtfA = MultiTermFit(omega_bestA, 10)
    mtfA.fit(tA, yA, dyA)
    phase_fitA, y_fitA, phased_tA = mtfA.predict(1000, return_phased_times=True)
    
    plt.errorbar(phased_tA, yA, dyA,
                     fmt='.r', ecolor='gray',
                     lw=1, ms=4, capsize=1.5)
    plt.plot(phase_fitA, y_fitA, '-b', lw=2, label="P_A = %.2f day" %(2 * np.pi / omega_bestA * 24. * 365.))
    ## B ##
    # get the data and best-fit angular frequency
    tB = data['t_eval'][data['window_id']==window]
    yB = data['sig_evalB'][data['window_id']==window]
    yB = (yB - np.mean(yB)) / np.std(yB)
    dyB = data['sig_errB'][data['window_id']==window]
    dyB = (dyB - np.mean(dyB)) / np.std(dyB)
    omegaB, powerB = resultsB[window]
    omega_bestB = omegaB[np.argmax(powerB)]
    print " - omega_0B = %.10g" % omega_bestB
    # do a fit to the first 10 Fourier components
    mtfB = MultiTermFit(omega_bestB, 10)
    mtfB.fit(tB, yB, dyB)
    phase_fitB, y_fitB, phased_tB = mtfB.predict(1000, return_phased_times=True)
    plt.errorbar(phased_tB, yB, dyB,
                     fmt='.k', ecolor='gray',
                     lw=1, ms=4, capsize=1.5)
    plt.plot(phase_fitB, y_fitB, '-c', lw=2, label="P_B = %.2f day" %(2 * np.pi / omega_bestB * 24. * 365.))
    plt.legend()
    plt.show()
