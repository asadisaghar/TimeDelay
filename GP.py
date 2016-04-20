# Split light curve into windows, build a GP model for each window,
# then correlate the GP models pair-wize for corresponding windows
# from the two light curves

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from sklearn.gaussian_process import GaussianProcess
from timedelay.gp import *
from timedelay.sampling_windows import *
from timedelay.correlation import *
from timedelay.plotting import *

do_plot = True
N_eval = 5000
dt = 0.1 #day
data = np.load("TimeDelayData/pairs_with_truths_and_windows.npz")['arr_0']
pair_ids = np.unique(data['full_pair_id'])[:1] # Remove [:5] for real run

# Fit GP model to each sampling window of the data separately
dt_preds = []

for pair_id in pair_ids:
    pair_data = data[data['full_pair_id'] == pair_id]

    tau = pair_data[0]['tau']
    sig = pair_data[0]['sig']
    dt_true = pair_data[0]['dt']

    print "loading data from pair %s" % pair_id

    windows = np.unique(pair_data['window_id'])

    modelA = np.zeros((N_eval * len(windows), 2))
    modelB = np.zeros((N_eval * len(windows), 2))
    t_eval = np.zeros((N_eval * len(windows), 1))

    if do_plot:
        figA, axsA = plt.subplots(len(windows), 1, figsize=(5, 10), sharey=True)#, sharey=True)
        axsA = axsA.flatten()
        figB, axsB = plt.subplots(len(windows), 1, figsize=(5, 10), sharey=True)#, sharey=True)
        axsB = axsB.flatten()

    for i, window in enumerate(windows):
        print "window %s..." % window
    
        window_data = pair_data[pair_data['window_id'] == window]
        t_eval[N_eval*i:N_eval*(i+1)] = np.atleast_2d(np.linspace(np.min(window_data['time']), np.max(window_data['time']), N_eval)).T
    
        # find the best-fit model for LC A
        XA = window_data['time'].T
        XA = XA.reshape((len(XA), 1))
        fA = (window_data['lcA'] - np.mean(window_data['lcA'])) / np.std(window_data['lcA'])
        dfA = (window_data['errA'] - np.mean(window_data['errA'])) / np.std(window_data['errA'])
        # modelA[N_eval*i:N_eval*(i+1), 0], modelA[N_eval*i:N_eval*(i+1), 1] = make_a_model(pair_id, 
        #                                                                                   t_eval[N_eval*i:N_eval*(i+1)], 
        #                                                                                   XA, fA, dfA,
        #                                                                                   theta0=sig,
        #                                                                                   thetaL=tau,
        #                                                                                   thetaU=tau) # image A

        # find the best-fit model for LC B
        XB = window_data['time'].T
        XB = XB.reshape((len(XB), 1))
        fB = (window_data['lcB'] - np.mean(window_data['lcB'])) / np.std(window_data['lcB'])
        dfB = (window_data['errB'] - np.mean(window_data['errB'])) / np.std(window_data['errB'])
        # modelB[N_eval*i:N_eval*(i+1), 0], modelB[N_eval*i:N_eval*(i+1), 1] = make_a_model(pair_id, 
        #                                                                                   t_eval[N_eval*i:N_eval*(i+1)], 
        #                                                                                   XB, fB, dfB,
        #                                                                                   theta0=sig,
        #                                                                                   thetaL=tau,
        #                                                                                   thetaU=tau) # image B

        # # Cross correlate the two models
        # ind_maxcorr, corr = cross_correlate_models(modelA[N_eval*i:N_eval*(i+1), 0],
        #                                            modelB[N_eval*i:N_eval*(i+1), 0])

        # maxcorr = t_eval[ind_maxcorr]
        # if corr[ind_maxcorr] < 0:
        #     t_maxcorr[i] = -maxcorr[0][0]
        # else:
        #     t_maxcorr[i] = maxcorr[0][0]
    
        ###  plot_model(pair_id, dt, x, y_pred, sigma, ob)   
        if do_plot:
            axsA[i] = plot_data(axsA[i], pair_id, 0, XA, fA, dfA, "A") #A
            # axsA[i] = plot_model(axsA[i], pair_id, 0, 
            #                      t_eval[N_eval*i:N_eval*(i+1)], 
            #                      modelA[N_eval*i:N_eval*(i+1), 0], 
            #                      modelA[N_eval*i:N_eval*(i+1), 1],
            #                      "A ") #A
    
            axsB[i] = plot_data(axsB[i], pair_id, 0, XB, fB, dfB, "B") #B
            # axsB[i] = plot_model(axsB[i], pair_id, 0, 
            #                      t_eval[N_eval*i:N_eval*(i+1)], 
            #                      modelB[N_eval*i:N_eval*(i+1), 0], 
            #                      modelB[N_eval*i:N_eval*(i+1), 1],
            #                      "B ") #B
    
    # dt_preds.append(np.mean(t_maxcorr))
    
    # if do_plot:
        # figA.suptitle("Lightcurve A \n Estimated time delay: " + 
        #               str(np.mean(t_maxcorr)) + 
        #               " days")
        # figB.suptitle("Lightcurve B \n True time delay: " + 
        #               str(dt_true) + 
        #               " days")
        # figA.savefig("pair" + str(pair_id) + "_A.png")
        # figB.savefig("pair" + str(pair_id) + "_B.png")

#np.save("dt_preds", dt_preds)
plt.show()


