import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from timedelay.plotting import *
from matplotlib import rcParams

rcParams['xtick.major.size'] = 4
rcParams['xtick.major.width'] = 2
rcParams['xtick.minor.size'] = 2
rcParams['xtick.minor.width'] = 2
rcParams['xtick.labelsize'] = 15

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

def normalize_sig(sig):
    return (sig - np.mean(sig)) / np.max(sig)

def make_fig_1(pair_id=120350):
    alpha = 0.9
    data = np.load("TimeDelayData/pairs_with_truths_and_windows.npz")['arr_0']
    data = data[data['full_pair_id'] == pair_id]
    true_dt = np.unique(data['dt'])
    fig, axs = plt.subplots(3, 1, figsize=(20, 10), sharex=True, sharey=False)
    axs[0].errorbar(data['time'], data['lcA'], data['errA'], fmt='.c')
    axs[0].set_title('Lightcurve A', color='c')
    axs[0].set_ylim(0, (np.max(data['lcA']) + np.max(data['errA'])))
    axs[1].errorbar(data['time'], data['lcB'], data['errB'], fmt='.m')
    axs[1].set_title('Lightcurve B', color='m')
    axs[1].set_ylim(0, (np.max(data['lcB']) + np.max(data['errB'])))
    axs[2].scatter(data['time'], normalize_sig(data['lcA']), marker='o', c='c', edgecolor='None', s=20)
    axs[2].scatter(data['time'], normalize_sig(data['lcB']), marker='o', c='m', edgecolor='None', s=20)
    axs[2].set_title('true time delay : %.2f days'%(true_dt))
    axs[2].set_ylim(-1, 1)
    plt.xlim(-100, 1600)
    plt.xlabel('time (days)')
    axs[0].set_ylabel('flux (nanomaggies)')
    axs[1].set_ylabel('flux (nanomaggies)')
    axs[2].set_ylabel('flux (normalized)')

    if __name__ == "__main__":
        plt.savefig('Report/Figures/Fig1.jpg')
    else:
        plt.show()

def make_fig_2(pair_id=120350):
    gp_modeled_data = np.load("TimeDelayData/gp_resampled_of_windows_with_truth.npz")['arr_0']
    raw_data = np.load("TimeDelayData/pairs_with_truths_and_windows.npz")['arr_0']
    pair_data = raw_data[raw_data['full_pair_id'] == pair_id]
    windows = np.unique(pair_data['window_id'])
    pair_model = gp_modeled_data[gp_modeled_data['pair_id'] == pair_id]
    gp_modeled_data = None
    raw_data = None
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
        axs[i,0].set_ylabel('flux (normalized)')            
    axs[0,0].set_title("Lightcurve A") 
    axs[0,1].set_title("Lightcurve B")
    axs[i,0].set_xlabel('time (days)')
    axs[i,1].set_xlabel('time (days)')

    if __name__ == "__main__":
        plt.savefig('Report/Figures/Fig2.jpg')
    else:
        plt.show()


def make_fig_3(pair_id=120350):
    data_corr = np.load("TimeDelayData/timeshift_correlate_normalized_detrend.npz")['arr_0']
    data_corr = data_corr[data_corr['pair_id'] == pair_id]
    data_mse = np.load("TimeDelayData/timeshift_mse_normalized_detrend.npz")['arr_0']
    data_mse = data_mse[data_mse['pair_id'] == pair_id]
    # mean_corr = np.zeros(len(data_corr))
    # mean_mse = np.zeros(len(data_mse))
    # w_mean_corr = np.zeros(len(data_corr))
    # w_mean_mse = np.zeros(len(data_mse))
    windows = np.unique(data_corr['window_id'])
    print windows
#    windows = np.unique(pair_mse['window_id'])
    fig, axs = plt.subplots(len(windows), 1, figsize=(20, 20), sharex=True, sharey=True)
    for i, window in enumerate(windows):
        window_corr = data_corr[data_corr['window_id'] == window]
        window_mse = data_mse[data_mse['window_id'] == window]
        axs[i].plot(window_corr['offset'], window_corr['correlation'] / np.max(window_corr['correlation']), 
                    '-', c='#980043', label='correlation')
        axs[i].plot(window_mse['offset'], window_mse['correlation'] / np.max(window_mse['correlation']),
                    '-', c='#c994c7', label='MSE') 
        axs[i].vlines(x=window_corr['dt'][0], ymin=np.min(window_corr['offset']), ymax=np.max(window_corr['offset']), 
                      colors='k', linestyle='solid')
            # mean_corr += window_corr['correlation']
            # mean_mse += window_mse['correlation']
            # w_mean_corr += (window_corr['correlation'] / np.max(window_corr['correlation'])) 
            # w_mean_mse += (window_mse['correlation'] / np.max(window_corr['correlation']))

    # axs[i+1].plot(window_mse['offset'], mean_corr / len(windows),
    #               '-', c='#980043', label='correlation') 
    # axs[i+1].plot(window_mse['offset'], w_mean_corr / len(windows),
    #               '--', c='#980043', label='correlation') 
    # axs[i+1].plot(window_mse['offset'], mean_corr / len(windows),
    #               '-', c='#c994c7', label='MSE') 
    # axs[i+1].plot(window_mse['offset'], w_mean_corr / len(windows),
    #               '--', c='#c994c7', label='MSE') 
    plt.xlabel('timeshift (days)')
    plt.ylabel('cost value')
    axs[0].legend()

    if __name__ == "__main__":
        plt.savefig('Report/Figures/Fig3.jpg')
    else:
        plt.show()


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
    plt.xlabel('true dt (days)')
    plt.ylabel('median timeshift (days)')
    plt.scatter(troubled_data['dt'], -troubled_data['est_dt_median'], 
             marker='o', facecolor='k', edgecolor='None', alpha=0.9, 
                label='troubled_data: %s windows'%(len(troubled_data)))
    plt.legend()

    if __name__ == "__main__":
        plt.savefig('Report/Figures/Fig4.jpg')
    else:
        plt.show()

    # mod_data = np.zeros((len(normal_data),))
    # for i in range(len(normal_data)):
    #     if normal_data['est_dt_median'][i]<0:
    #         mod_data[i] = normal_data['est_dt_median'][i]-normal_data['est_dt_std'][i]
    #     else:
    #         mod_data[i] = normal_data['est_dt_median'][i]+normal_data['est_dt_std'][i]
    #         plt.plot(np.abs(normal_data['dt']), np.abs(mod_data), 'ok', lw=0, alpha=0.3)
    #         plt.show()

if __name__ == "__main__":
    #make_fig_1()
#    make_fig_2()
    #make_fig_3()
    make_fig_4()
