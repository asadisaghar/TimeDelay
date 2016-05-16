import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from timedelay.plotting import *
from matplotlib import rcParams

rcParams['xtick.major.size'] = 3
rcParams['xtick.major.width'] = 3
rcParams['xtick.minor.size'] = 2
rcParams['xtick.minor.width'] = 2
rcParams['xtick.labelsize'] = 15

rcParams['ytick.major.size'] = rcParams['xtick.major.size']
rcParams['ytick.major.width'] = rcParams['xtick.major.width']
rcParams['ytick.minor.size'] = rcParams['xtick.minor.size']
rcParams['ytick.minor.width'] = rcParams['xtick.minor.width']
rcParams['ytick.labelsize'] = rcParams['xtick.labelsize']

rcParams['lines.linewidth'] = 3

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
    fig, axs = plt.subplots(3, 1, figsize=(8, 13), sharex=True, sharey=False)
    axs[0].errorbar(data['time'], data['lcA'], data['errA'], fmt='oc')
    axs[0].set_title('Lightcurve A', color='c')
    axs[0].set_ylim(0, (np.max(data['lcA']) + np.max(data['errA'])))
    axs[1].errorbar(data['time'], data['lcB'], data['errB'], fmt='o', c='darkorange')
    axs[1].set_title('Lightcurve B', color='darkorange')
    axs[1].set_ylim(0, (np.max(data['lcB']) + np.max(data['errB'])))
    axs[2].scatter(data['time'], normalize_sig(data['lcA']), marker='o', c='c', edgecolor='None', s=40, alpha=0.6)
    axs[2].scatter(data['time'], normalize_sig(data['lcB']), marker='o', c='darkorange', edgecolor='None', s=40, alpha=0.6)
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
    fig, axs = plt.subplots(len(windows), 2, figsize=(8, 13), sharey=True)

    for i, window in enumerate(windows):
        try:
            window_data = pair_data[pair_data['window_id'] == window]
            window_model = pair_model[pair_model['window_id'] == window]

            XA = window_data['time'].T
            XA = XA.reshape((len(XA), 1))
            fA = (window_data['lcA'] - np.mean(window_data['lcA'])) / np.std(window_data['lcA'])
            dfA = (window_data['errA'])# - np.mean(window_data['errA'])) / np.std(window_data['errA'])

            XB = window_data['time'].T
            XB = XB.reshape((len(XB), 1))
            fB = (window_data['lcB'] - np.mean(window_data['lcB'])) / np.std(window_data['lcB'])
            dfB = (window_data['errB'])# - np.mean(window_data['errB'])) / np.std(window_data['errB'])

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


def make_fig_3(pair_id=120300):
    data_corr = np.load("TimeDelayData/timeshift_correlate_normalized_detrend.npz")['arr_0']
    data_corr = data_corr[data_corr['pair_id'] == pair_id]
    data_mse = np.load("TimeDelayData/timeshift_mse_normalized_detrend.npz")['arr_0']
    data_mse = data_mse[data_mse['pair_id'] == pair_id]
    windows = np.unique(data_corr['window_id'])
    max_corrs = np.zeros(len(windows))
    min_mses = np.zeros(len(windows))
    fig, axs = plt.subplots(len(windows), 1, figsize=(10, 15), sharex=True, sharey=True)
    for i, window in enumerate(windows):
        window_corr = data_corr[data_corr['window_id'] == window]
        window_mse = data_mse[data_mse['window_id'] == window]
        axs[i].plot(window_corr['offset'], window_corr['correlation'] / np.max(np.abs(window_corr['correlation'])), 
                    '-', c='c', label='correlation')
        axs[i].plot(window_mse['offset'], window_mse['correlation'] / np.max(np.abs(window_mse['correlation'])),
                    '-', c='darkorange', label='MSE') 
        axs[i].vlines(x=-window_corr['dt'][0], ymin=-1, ymax=1, 
                      colors='k', linestyle='solid', label='true dt')
        axs[i].set_ylabel('cost value')
        axs[i].set_ylim(-1.0, 1.0)
        max_corrs[i] = window_corr['offset'][np.argmax(window_corr['correlation'])]
        min_mses[i] = -window_mse['offset'][np.argmin(window_mse['correlation'])]

    plt.xlabel('timeshift (days)')
    median_corrs = np.median(max_corrs)
    std_corrs = np.abs(np.std(max_corrs))
    median_mses = np.median(max_corrs)
    std_mses = np.abs(np.std(min_mses))
    # print "corrs", max_corrs
    # print median_corrs, std_corrs
    # print "MSEs", min_mses
    # print median_mses, std_mses

    for i, window in enumerate(windows):
        axs[i].axvspan(xmin=(median_corrs - std_corrs), xmax=(median_corrs + std_corrs), facecolor='c', alpha=0.3)
        axs[i].axvspan(xmin=(median_mses - std_mses), xmax=(median_mses + std_mses), facecolor='darkorange', alpha=0.3)

        axs[i].vlines(x=median_corrs, ymin=-1, ymax=1, 
                      colors='c', linestyle='solid')
        axs[i].vlines(x=median_mses, ymin=-1, ymax=1, 
                      colors='darkorange', linestyle='solid')

    axs[0].legend()

    if __name__ == "__main__":
        plt.savefig('Report/Figures/Fig3.jpg')
    else:
        plt.show()

def make_fig_4():
    def norm(a):                                                                                                              
        return (a - a.mean()) / a.std()
    data = np.load("TimeDelayData/timeshift_correlate_normalized_detrend.measures.pairsums.npz")['arr_0'] #troubled-data re excluded from this dataset already, so some of the lines below are just redundant leftovers!
    x = np.load("TimeDelayData/timeshift_correlate_normalized_detrend.measures.pairsums.clusters.npz")['arr_0']                               
    troubled_data = data[data['est_dt_median'] == 0]
    normal_data = np.setdiff1d(data, troubled_data)
    fig, axs = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True)
    axs.scatter(normal_data['dt'], -(normal_data['est_dt_median'] + normal_data['est_dt_std'] * np.sign(normal_data['est_dt_median'])), 
                 marker='o', facecolor='c', edgecolor='None', alpha=0.5, 
                 label='dt median w/o clustering')
    axs.plot(normal_data['dt'], normal_data['dt'], 
             '-', c='gray')
    axs.set_xlabel('true dt (days)')
    axs.set_ylabel('median timeshift (days)')
    axs.set_xlim(-150, 150)
    axs.set_ylim(-150, 150)
    axs.scatter(x['dt'], -((x['est_dt_median'] + x['est_dt_std'] * np.sign(x['est_dt_median']))), 
                marker='o', facecolor='darkorange', edgecolor='None', s=40, 
                label='dt median w/ clustering')
    axs.legend()
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

def make_fig_5():
    data = np.load("TimeDelayData/timeshift_correlate_normalized_detrend.measures.pairsums.npz")['arr_0']
    fig, axs = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True)
    axs.scatter(data['dt'], -data['est_dt_median'], 
                 marker='o', facecolor='c', edgecolor='None', alpha=0.5, label='dt median')
    axs.scatter(data['dt'], data['est_dt_std'], 
                 marker='o', facecolor='darkorange', edgecolor='None', s=30, alpha=0.5, label='dt stand dard deviation across windows')
    axs.plot(data['dt'], data['dt'],
             '-', c='gray')
    axs.set_xlabel('true dt (days)')
    axs.set_xlim(-150, 150)
    axs.set_ylim(-150, 150)
    axs.legend()
    if __name__ == "__main__":
        plt.savefig('Report/Figures/Fig5.jpg')
    else:
        plt.show()

def make_fig_6():
    reg = np.load("TimeDelayData/timeshift_correlate_normalized_detrend.measures.pairsums.regression-ests.npz")['arr_0']
    heur = np.load("TimeDelayData/timeshift_correlate_normalized_detrend.measures.pairsums.ests.npz")['arr_0']

    fig, axs = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True)
    axs.hist(reg['est'] - reg['dt'], bins=50, normed=True, 
             histtype='step', color='darkorange', label='Regression')
    axs.hist(heur['est'] - heur['dt'], bins=50, normed=True,
             histtype='step', color='c', label='Heuristic')
    axs.legend()

    if __name__ == "__main__":
        plt.savefig('Report/Figures/Fig6.jpg')
    else:
        plt.show()



if __name__ == "__main__":
#    make_fig_1(pair_id=120354)
#    make_fig_2(pair_id=120354)
#    make_fig_3(pair_id=120354) #3.1
#    make_fig_3(pair_id=120300) #3.2
#    make_fig_4()
#    make_fig_6()

