import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from timedelay.plotting import *
from matplotlib import rcParams

rcParams['xtick.major.size'] = 4
rcParams['xtick.major.width'] = 2
rcParams['xtick.minor.size'] = 2
rcParams['xtick.minor.width'] = 2
rcParams['xtick.labelsize'] = 20

rcParams['ytick.major.size'] = rcParams['xtick.major.size']
rcParams['ytick.major.width'] = rcParams['xtick.major.width']
rcParams['ytick.minor.size'] = rcParams['xtick.minor.size']
rcParams['ytick.minor.width'] = rcParams['xtick.minor.width']
rcParams['ytick.labelsize'] = rcParams['xtick.labelsize']

rcParams['lines.linewidth'] = 2

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
rcParams['font.size'] = 20
rcParams['text.usetex'] = True


def make_fig_1(pair_id=120350):
    alpha = 0.9
    data = np.load("TimeDelayData/pairs_with_truths_and_windows.npz")['arr_0']
#    pair_id = 120258
    pair_data = data[data['full_pair_id'] == pair_id]
    true_dt = np.unique(pair_data['dt'])
    fig, axs = plt.subplots(3, 1, figsize=(20, 10), sharex=True, sharey=False)
    axs[0].errorbar(pair_data['time'], pair_data['lcA'], pair_data['errA'], fmt='.c')
    axs[0].set_title('Lightcurve A', color='c')
#    axs[0].set_ylim(0, 1.0)
    axs[1].errorbar(pair_data['time'], pair_data['lcB'], pair_data['errB'], fmt='.m')
    axs[1].set_title('Lightcurve B', color='m')
#    axs[1].set_ylim(0, 1.5)
    axs[2].scatter(pair_data['time'], normalize_sig(pair_data['lcA']), marker='o', c='c', edgecolor='None', s=20)
    axs[2].scatter(pair_data['time'], normalize_sig(pair_data['lcB']), marker='o', c='m', edgecolor='None', s=20)
    axs[2].set_title('true time delay : %.2f days'%(true_dt))
    axs[2].set_ylim(-5, 5)
    plt.xlim(-100, 1600)
    plt.show()

def normalize_sig(sig):
    return (sig - np.mean(sig)) / np.std(sig)

def make_fig_2(pair_id=120350):
    #tdc = 1
    #rungs  = np.arange(0, 1)
    #pair = 17
    #full_pair_ids = pair + rungs * 10000 + tdc * 100000
    gp_modeled_data = np.load("TimeDelayData/gp_resampled_of_windows_with_truth.npz")['arr_0']
#    full_pair_ids = np.unique(gp_modeled_data['pair_id'][gp_modeled_data['pair_id']==120258])
    full_pair_ids = [pair_id]
    raw_data = np.load("TimeDelayData/pairs_with_truths_and_windows.npz")['arr_0']
    for pair_id in full_pair_ids:
        pair_data = raw_data[raw_data['full_pair_id'] == pair_id]
        windows = np.unique(pair_data['window_id'])
        pair_model = gp_modeled_data[gp_modeled_data['pair_id'] == pair_id]
        fig, axs = plt.subplots(len(windows), 2, figsize=(20, 20), sharey=True)

    for i, window in enumerate(windows):
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
                              np.sqrt(window_model['sig_errA']),
                             "A ") #A
    
        axs[i,1] = plot_data(axs[i,1], pair_id, 0, XB, fB, dfB, "B") #B
        axs[i,1] = plot_model(axs[i,1], pair_id, 0, 
                              window_model['t_eval'],
                              window_model['sig_evalB'],
                              np.sqrt(window_model['sig_errB']),
                             "B ") #B
            
    axs[0,0].set_title("Lightcurve A") 
    axs[0,1].set_title("Lightcurve B")
    plt.show()

def make_fig_3(pair_id=120350):
    data_corr = np.load("TimeDelayData/timeshift_correlate_normalized_detrend.npz")['arr_0']
    data_mse = np.load("TimeDelayData/timeshift_mse_normalized_detrend.npz")['arr_0']
    pair_corr = data_corr[data_corr['pair_id'] == pair_id]
    pair_mse = data_mse[data_mse['pair_id'] == pair_id]
    windows = np.unique(pair_corr['window_id'])
#    windows = np.unique(pair_mse['window_id'])
    fig, axs = plt.subplots(len(windows), 1, figsize=(20, 20), sharex=True, sharey=True)
    for i, window in enumerate(windows):
        try:
            window_corr = pair_corr[pair_corr['window_id'] == window]
            window_mse = pair_mse[pair_mse['window_id'] == window]
            axs[i].plot(window_corr['offset'], window_corr['correlation'] / np.max(window_corr['correlation']), 
                        '-', c='#980043', label='correlation')
            axs[i].plot(window_mse['offset'], window_mse['correlation'] / window_mse['correlation'],
                        '-', c='#c994c7', label='MSE') 
            plt.legend()
        except Exception, e:
            print "    ", e

    plt.savefig("corr_vs_mse.jpg")
#    plt.show()

def make_fig_4():
    data = np.load("TimeDelayData/dt_correlate_normalized_wgtd.npz")['arr_0']
    troubled_data = data[data['est_dt_median'] == 0]
    normal_data = np.setdiff1d(data, troubled_data)
    fig = plt.figure(figsize=(20,20))
    plt.errorbar(normal_data['dt'], -normal_data['est_dt_median'], 
                 yerr=np.sqrt(2. * normal_data['est_dt_std']), 
                 fmt='.c', alpha=0.5)
    plt.plot(normal_data['dt'], -normal_data['est_dt_median'], 
             '.b', label='normal data : %s windows'%(len(normal_data)))
    plt.plot(normal_data['dt'], normal_data['dt'], 
             '-r', alpha=0.8)

    yminmax = np.min(np.abs(data['dt']))
    plt.axhspan(ymin=-yminmax, ymax=yminmax, facecolor='r', alpha=0.5)
    plt.xlabel('dt_true (days)')
    plt.ylabel('dt_sindow median (days)')
    plt.scatter(troubled_data['dt'], -troubled_data['est_dt_median'], 
             marker='o', facecolor='k', edgecolor='None', alpha=0.9, 
                label='troubled_data: %s windows'%(len(troubled_data)))
    plt.legend()
    plt.show()
    # mod_data = np.zeros((len(normal_data),))
    # for i in range(len(normal_data)):
    #     if normal_data['est_dt_median'][i]<0:
    #         mod_data[i] = normal_data['est_dt_median'][i]-normal_data['est_dt_std'][i]
    #     else:
    #         mod_data[i] = normal_data['est_dt_median'][i]+normal_data['est_dt_std'][i]
    #         plt.plot(np.abs(normal_data['dt']), np.abs(mod_data), 'ok', lw=0, alpha=0.3)
    #         plt.show()

# make_fig_1()
# make_fig_2()
make_fig_3()
#make_fig_4()
