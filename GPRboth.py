# Merge all the observations of light curve A and B, with observations
# from B offset by the dt from the ground truth table. Then fit a GP
# model to the resulting dataset and plot it.

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from sklearn.gaussian_process import GaussianProcess
plt.set_cmap('nipy_spectral') 
from timedelay.gp import *
from timedelay.sampling_windows import *
from timedelay.correlation import *
from timedelay.plotting import *

# Fit GP model to each sampling window of the data separately
N_eval = 25000
path = "tdc1/rung3/"
truth = np.loadtxt(path + "truth3.txt", skiprows=1,
                   dtype={"names":("pairfile", "dt", "m1", "m2", "zl", "zs", "id", "tau", "sig"),
                          "formats":("S30", "f4", "f4", "f4", "f4", "f4", "f4", "f4", "f4")})
pairNos = np.linspace(1, 2, 1)
dt_trues = truth["dt"]
taus = truth["tau"]
sigs = truth["sig"]

for pairNo, tau, sig, dt_true in zip(pairNos, taus, sigs, dt_trues):
    print "loading data from pair" + str(pairNo)
    lightcurve = np.loadtxt(path + "tdc1_rung3_double_pair%d.txt"%(pairNo),
                            skiprows=6, dtype={"names":("time", "lcA", "errA", "lcB", "errB"), 
                                               "formats":("f4", "f4", "f4", "f4", "f4")})

    x = np.atleast_2d(np.linspace(0, np.max(lightcurve['time']), 5000)).T
    XA = lightcurve['time'].T
    XB = (lightcurve['time'] + dt_true).T
    X = np.concatenate((XA, XB))
    X = X.reshape((len(X), 1))
    yA = (lightcurve['lcA'] - np.mean(lightcurve['lcA'])) / np.std(lightcurve['lcA'])
    yB = (lightcurve['lcB'] - np.mean(lightcurve['lcB'])) / np.std(lightcurve['lcB'])
    y = np.concatenate((yA, yB))
    dyA = (lightcurve['errA'] - np.mean(lightcurve['errA'])) / np.std(lightcurve['errA'])
    dyB = (lightcurve['errB'] - np.mean(lightcurve['errB'])) / np.std(lightcurve['errB'])
    dy = np.concatenate((dyA, dyB))

    y_pred, sigma = make_a_model(pairNo, x, X, y, dy, theta0=sig, thetaL=tau, thetaU=tau) #A
    
    print "Pair " + str(pairNo) + " done!"
    
    # Plot everything
    fig = plt.figure()
    ax = fig.add_subplot(111)
    axA = plot_data(ax, pairNo, 0, X, y, dy, "A and B")
    ax = plot_model(ax, pairNo, 0, x, y_pred, sigma,"A and B")

    
    plt.xlabel('t [days]')
    plt.ylabel('normalized flux [arbitrary]')
    fig.suptitle('PairNo: ' + str(pairNo))

    
    plt.show()
