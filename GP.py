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
import cPickle as pickle

do_plot = True
N_eval = 500
dt = 0.1 #day
data = np.load("TimeDelayData/pairs_with_truths_and_windows.npz")['arr_0'][:500]
pair_ids = np.unique(data['full_pair_id']) # Remove [:5] for real run
print len(pair_ids)
quad_pair_ids = pair_ids[pair_ids % 1.0 == 0.5] # Finde file A of a quadratic system
print len(quad_pair_ids)
quad_pair_ids = np.append(quad_pair_ids, quad_pair_ids - 0.5) # Finds file B of the same quadratic system
print len(quad_pair_ids)
pair_ids = np.setdiff1d(pair_ids, quad_pair_ids) # Removes all quadratic systems from pair_ids
# Fit GP model to each sampling window of the data separately

for pair_id in pair_ids:
    pair_data = data[data['full_pair_id'] == pair_id]

    tau = pair_data[0]['tau']
    sig = pair_data[0]['sig']

    print "loading data from pair %s" % pair_id

    windows = np.unique(pair_data['window_id'])

    modelA = np.zeros((N_eval * len(windows), 2))
    modelB = np.zeros((N_eval * len(windows), 2))
    t_eval = np.zeros((N_eval * len(windows), 1))

    if do_plot:
        figA, axsA = plt.subplots(len(windows), 1, figsize=(5, 10), sharey=True)
        axsA = axsA.flatten()
        figB, axsB = plt.subplots(len(windows), 1, figsize=(5, 10), sharey=True)
        axsB = axsB.flatten()

    for i, window in enumerate(windows):
        print "window %s..." % window
    
        try:
            window_data = pair_data[pair_data['window_id'] == window]
            t_eval[N_eval*i:N_eval*(i+1)] = np.atleast_2d(np.linspace(np.min(window_data['time']), np.max(window_data['time']), N_eval)).T

            # find the best-fit model for LC A
            XA = window_data['time'].T
            XA = XA.reshape((len(XA), 1))
            fA = (window_data['lcA'] - np.mean(window_data['lcA'])) / np.std(window_data['lcA'])
            dfA = (window_data['errA'] - np.mean(window_data['errA'])) / np.std(window_data['errA'])
            gpA, modelA[N_eval*i:N_eval*(i+1), 0], modelA[N_eval*i:N_eval*(i+1), 1] = make_a_model(pair_id, 
                                                                                                   t_eval[N_eval*i:N_eval*(i+1)], 
                                                                                                   XA, fA, dfA,
                                                                                                   theta0=sig,
                                                                                                   thetaL=tau,
                                                                                                   thetaU=tau) # image A

            with open("GPModels/%sA.pkl"%(window), "wb") as f:
                pickle.dump(gpA, f)

            # find the best-fit model for LC B
            XB = window_data['time'].T
            XB = XB.reshape((len(XB), 1))
            fB = (window_data['lcB'] - np.mean(window_data['lcB'])) / np.std(window_data['lcB'])
            dfB = (window_data['errB'] - np.mean(window_data['errB'])) / np.std(window_data['errB'])
            gpB, modelB[N_eval*i:N_eval*(i+1), 0], modelB[N_eval*i:N_eval*(i+1), 1] = make_a_model(pair_id, 
                                                                                                   t_eval[N_eval*i:N_eval*(i+1)], 
                                                                                                   XB, fB, dfB,
                                                                                                   theta0=sig,
                                                                                                   thetaL=tau,
                                                                                                   thetaU=tau) # image B
            with open("GPModels/%sB.pkl"%(window), "wb") as f:
                pickle.dump(gpB, f)
        except Exception, e:
            print "    ", e

    
        ###  plot_model(pair_id, dt, x, y_pred, sigma, ob)   
        if do_plot:
            axsA[i] = plot_data(axsA[i], pair_id, 0, XA, fA, dfA, "A") #A
            axsA[i] = plot_model(axsA[i], pair_id, 0, 
                                 t_eval[N_eval*i:N_eval*(i+1)], 
                                 modelA[N_eval*i:N_eval*(i+1), 0], 
                                 modelA[N_eval*i:N_eval*(i+1), 1],
                                 "A ") #A
    
            axsB[i] = plot_data(axsB[i], pair_id, 0, XB, fB, dfB, "B") #B
            axsB[i] = plot_model(axsB[i], pair_id, 0, 
                                 t_eval[N_eval*i:N_eval*(i+1)], 
                                 modelB[N_eval*i:N_eval*(i+1), 0], 
                                 modelB[N_eval*i:N_eval*(i+1), 1],
                                 "B ") #B
            
    if do_plot:
        figA.suptitle("Lightcurve A") 
        figB.suptitle("Lightcurve B")
        figA.savefig("pair" + str(pair_id) + "_A.png")
        figB.savefig("pair" + str(pair_id) + "_B.png")

plt.show()


