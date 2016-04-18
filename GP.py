import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from sklearn.gaussian_process import GaussianProcess
from timedelay.gp import *
from timedelay.sampling_windows import *
from timedelay.correlation import *
from timedelay.plotting import *

do_plot = False

# Fit GP model to each sampling window of the data separately
N_eval = 5000
path = "tdc1/rung3/"
truth = np.loadtxt(path + "truth3.txt", skiprows=1,
                   dtype={"names":("pairfile", "dt", "m1", "m2", "zl", "zs", "id", "tau", "sig"),
                          "formats":("S30", "f4", "f4", "f4", "f4", "f4", "f4", "f4", "f4")})

pairNos = np.arange(1, 720, 1)
dt_trues = truth["dt"]
taus = truth["tau"]
sigs = truth["sig"]
dt_preds = []
for pairNo, tau, sig, dt_true in zip(pairNos, taus, sigs, dt_trues):
    print "loading data from pair" + str(pairNo)
    lightcurve = np.loadtxt(path + "tdc1_rung3_double_pair%d.txt"%(pairNo),
                            skiprows=6, dtype={"names":("time", "lcA", "errA", "lcB", "errB"), 
                                               "formats":("f4", "f4", "f4", "f4", "f4")})

    windows = pick_sampling_windows(lightcurve['time'])
    modelA = np.zeros((N_eval * len(windows), 2))
    modelB = np.zeros((N_eval * len(windows), 2))
    t_maxcorr = np.zeros((len(windows), ))
    t_eval = np.zeros((N_eval * len(windows), 1))
    periodA = np.zeros((len(windows), ))
    periodB = np.zeros((len(windows), ))

    if do_plot:
        figA, axsA = plt.subplots(len(windows), 1, figsize=(5, 10), sharey=True)#, sharey=True)
        axsA = axsA.flatten()
        figB, axsB = plt.subplots(len(windows), 1, figsize=(5, 10), sharey=True)#, sharey=True)
        axsB = axsB.flatten()

    for i, window in enumerate(windows):
        print "window " + str(i) + " ..."
    
        data = lightcurve[window[0]:window[1]]
        t_eval[N_eval*i:N_eval*(i+1)] = np.atleast_2d(np.linspace(np.min(data['time']), np.max(data['time']), N_eval)).T
    
        # find the best-fit model for LC A
        XA = data['time'].T
        XA = XA.reshape((len(XA), 1))
        fA = (data['lcA'] - np.mean(data['lcA'])) / np.std(data['lcA'])
        dfA = (data['errA'] - np.mean(data['errA'])) / np.std(data['errA'])
        modelA[N_eval*i:N_eval*(i+1), 0], modelA[N_eval*i:N_eval*(i+1), 1] = make_a_model(pairNo, 
                                                                                          t_eval[N_eval*i:N_eval*(i+1)], 
                                                                                          XA, fA, dfA,
                                                                                          theta0=sig,
                                                                                          thetaL=tau,
                                                                                          thetaU=tau) # image A

        # find the best-fit model for LC B
        XB = data['time'].T
        XB = XB.reshape((len(XB), 1))
        fB = (data['lcB'] - np.mean(data['lcB'])) / np.std(data['lcB'])
        dfB = (data['errB'] - np.mean(data['errB'])) / np.std(data['errB'])
        modelB[N_eval*i:N_eval*(i+1), 0], modelB[N_eval*i:N_eval*(i+1), 1] = make_a_model(pairNo, 
                                                                                          t_eval[N_eval*i:N_eval*(i+1)], 
                                                                                          XB, fB, dfB,
                                                                                          theta0=sig,
                                                                                          thetaL=tau,
                                                                                          thetaU=tau) # image B

        # Cross correlate the two models
        ind_maxcorr, corr = cross_correlate_models(modelA[N_eval*i:N_eval*(i+1), 0],
                                                   modelB[N_eval*i:N_eval*(i+1), 0])

        maxcorr = t_eval[ind_maxcorr]
        if corr[ind_maxcorr] < 0:
            t_maxcorr[i] = -maxcorr[0][0]
        else:
            t_maxcorr[i] = maxcorr[0][0]
    
        ###  plot_model(pairNo, dt, x, y_pred, sigma, ob)   
        if do_plot:
            axsA[i] = plot_data(axsA[i], pairNo, 0, XA, fA, dfA, "A") #A
            axsA[i] = plot_model(axsA[i], pairNo, 0, 
                                 t_eval[N_eval*i:N_eval*(i+1)], 
                                 modelA[N_eval*i:N_eval*(i+1), 0], 
                                 modelA[N_eval*i:N_eval*(i+1), 1],
                                 "A ") #A
    
            axsB[i] = plot_data(axsB[i], pairNo, 0, XB, fB, dfB, "B") #B
            axsB[i] = plot_model(axsB[i], pairNo, 0, 
                                 t_eval[N_eval*i:N_eval*(i+1)], 
                                 modelB[N_eval*i:N_eval*(i+1), 0], 
                                 modelB[N_eval*i:N_eval*(i+1), 1],
                                 "B ") #B
    
    dt_preds.append(np.mean(t_maxcorr))
    
    if do_plot:
        figA.suptitle("Lightcurve A \n Estimated time delay: " + 
                      str(np.mean(t_maxcorr)) + 
                      " days")
        figB.suptitle("Lightcurve B \n True time delay: " + 
                      str(dt_true) + 
                      " days")
        figA.savefig("pair" + str(pairNo) + "_A.png")
        figB.savefig("pair" + str(pairNo) + "_B.png")

np.save("dt_preds", dt_preds)
plt.show()


