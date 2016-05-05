import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from timedelay.plotting import *

def make_fig_1():
    fig, axs = plt.subplots(5, 2, figsize=(20, 15))
    alpha = 0.9
    fs = 20
    for i in range(0, 5):
        truth = np.loadtxt("tdc1/rung%d/truth%d.txt"%(i, i), 
                           skiprows=1,
                           dtype={"names":("pairfile", "dt", "m1", "m2", "zl", "zs", "id", "tau", "sig"),
                                  "formats":("S30", "f4", "f4", "f4", "f4", "f4", "f4", "f4", "f4")})
        dt_true = truth['dt'][truth['pairfile']=="tdc1_rung%d_double_pair1.txt"%(i)]
        lightcurves = np.loadtxt("tdc1/rung%d/tdc1_rung%d_double_pair1.txt"%(i, i),
                                 skiprows=6,
                                 dtype={"names":("time", "lcA", "errA", "lcB", "errB"),
                                        "formats":("f4", "f4", "f4", "f4", "f4")})
        t = lightcurves['time']
        A = lightcurves['lcA']
        normA = (A - np.mean(A)) / np.std(A)
        B = lightcurves['lcB']
        normB = (B - np.mean(B)) / np.std(B)
        axs[i,0].errorbar(t, A, lightcurves['errA'], fmt='.', c='#f03b20',  label='Light curve A', alpha=alpha)
        axs[i,0].errorbar(t, B, lightcurves['errB'], fmt='.', c='#2c7fb8',  label='Light curve B', alpha=alpha)

        axs[i,1].errorbar(t, normA, lightcurves['errA'], fmt='.', c='#f03b20',  label='Light curve A', alpha=alpha)
        axs[i,1].errorbar(t, normB, lightcurves['errB'], fmt='.', c='#2c7fb8',  label='Light curve B', alpha=alpha)

        # axs[i,0].scatter(t, A, marker='o', facecolor='#f03b20', edgecolor='None', label='Light curve A', alpha=alpha)
        # axs[i,0].scatter(t, B, marker='o', facecolor='#2c7fb8', edgecolor='None', label='Light curve B', alpha=alpha)
        # axs[i,1].scatter(t, normA, marker='o', facecolor='#f03b20', edgecolor='None', label='Light curve A', alpha=alpha)
        # axs[i,1].scatter(t, normB, marker='o', facecolor='#2c7fb8', edgecolor='None', label='Light curve B', alpha=alpha)
        axs[i,0].set_xlim(0, 3500)
        axs[i,0].set_title('rung%d - pair 1 - true dt = %.2f days'%(i, dt_true))
        axs[i,1].set_xlim(0, 3500)
        axs[i,1].set_title('rung%d - pair 1 - true dt = %.2f days'%(i, dt_true))
        axs[0,1].legend()

    axs[0,0].set_title('Raw light curves', fontsize=fs)
    axs[0,1].set_title('Centered & noramalized light curves', fontsize=fs)
    plt.show()

def normalize_sig(sig):
    return (sig - np.mean(sig)) / np.std(sig)

def make_fig_2():
    #tdc = 1
    #rungs  = np.arange(0, 1)
    #pair = 17
    #full_pair_ids = pair + rungs * 10000 + tdc * 100000
    gp_modeled_data = np.load("TimeDelayData/gp_resampled_of_windows_with_truth.npz")['arr_0']
    full_pair_ids = np.unique(gp_modeled_data['pair_id'])[99:100]
    raw_data = np.load("TimeDelayData/pairs_with_truths_and_windows.npz")['arr_0']
    for pair_id in full_pair_ids:
        pair_data = raw_data[raw_data['full_pair_id'] == pair_id]
        windows = np.unique(pair_data['window_id'])
        pair_model = gp_modeled_data[gp_modeled_data['pair_id'] == pair_id]
        fig, axs = plt.subplots(len(windows), 2, figsize=(20, 30), sharey=True)

    for i, window in enumerate(windows):
        print "window %s..." % window    
        try:
            window_data = pair_data[pair_data['window_id'] == window]
            window_model = pair_model[pair_model['window_id'] == window]

            XA = window_data['time'].T
            XA = XA.reshape((len(XA), 1))
            fA = (window_data['lcA'] - np.mean(window_data['lcA'])) / np.std(window_data['lcA'])
            dfA = (window_data['errA'] - np.mean(window_data['errA'])) / np.std(window_data['errA'])

            XB = window_data['time'].T
            XB = XB.reshape((len(XB), 1))
            fB = (window_data['lcB'] - np.mean(window_data['lcB'])) / np.std(window_data['lcB'])
            dfB = (window_data['errB'] - np.mean(window_data['errB'])) / np.std(window_data['errB'])

        except Exception, e:
            print "    ", e
    
        ###  plot_model(pair_id, dt, x, y_pred, sigma, ob)   
        axs[i,0] = plot_data(axs[i,0], pair_id, 0, XA, fA, dfA, "A") #A
        axs[i,0] = plot_model(axs[i,0], pair_id, 0, 
                              window_model['t_eval'],
                              window_model['sig_evalA'],
                              window_model['sig_errA'],
                             "A ") #A
    
        axs[i,1] = plot_data(axs[i,1], pair_id, 0, XB, fB, dfB, "B") #B
        axs[i,1] = plot_model(axs[i,1], pair_id, 0, 
                              window_model['t_eval'],
                              window_model['sig_evalB'],
                              window_model['sig_errB'],
                             "B ") #B
            
    axs[0,0].set_title("Lightcurve A") 
    axs[0,1].set_title("Lightcurve B")

    plt.show()










        # for i, window in enumerate(windows):
        #     window_data = pair_data[pair_data['window_id'] == window]
        #     window_model = pair_model[pair_model['window_id'] == window]
        #     axs[i,0].errorbar(window_data['time'], normalize_sig(window_data['lcA']), window_data['errA'], 
        #                       fmt='.c', markersize=10, label='Observations A')
        #     axs[i,1].errorbar(window_data['time'], normalize_sig(window_data['lcB']), window_data['errB'], 
        #                       fmt='.c', markersize=10, label='Observations B')
        #     axs[i,0].plot(window_model['t_eval'], normalize_sig(window_model['sig_evalA']), '-k', lw=2, label='Prediction A')
        #     axs[i,1].plot(window_model['t_eval'], normalize_sig(window_model['sig_evalB']), '-k', lw=2, label='Prediction B')
        #     axs[i,0].fill(np.concatenate([window_model['t_eval'], window_model['t_eval'][::-1]]),
        #                   np.concatenate([window_model['sig_evalA'] - 1.9600 * window_model['sig_errA'], 
        #                                   (window_model['sig_evalA'] + 1.9600 * window_model['sig_errA'])[::-1]]), 
        #                   alpha=0.5, color='darkorange', ec='None', label='95% confidence interval A') 
        #     axs[i,1].fill(np.concatenate([window_model['t_eval'], window_model['t_eval'][::-1]]), 
        #                   np.concatenate([window_model['sig_evalB'] - 1.9600 * window_model['sig_errB'], 
        #                                   (window_model['sig_evalB'] + 1.9600 * window_model['sig_errB'])[::-1]]), 
        #                   alpha=0.5, color='darkorange', ec='None', label='95% confidence interval B') 
        # plt.show()
