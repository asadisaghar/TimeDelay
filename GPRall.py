import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from sklearn.gaussian_process import GaussianProcess
plt.set_cmap('nipy_spectral') 

## TOOLS ##
###########

# Find border indices of sampling windows in data
def pick_sampling_windows(timestamps):
    dt = []
    for i in xrange(1, len(timestamps)):
        dt.append(timestamps[i]-timestamps[i-1])
    dt_threshold = np.mean(dt)*1.
    right_ends = np.append(np.where(dt>=dt_threshold)[0], len(timestamps)-1)
    left_ends = np.append(0, np.where(dt>=dt_threshold)[0]+1)
    windows = np.zeros((len(right_ends), 2))
    for i in range(0, len(right_ends)):
        windows[i] = (left_ends[i], right_ends[i])
    return windows

# Make a GaussianProcess model for noisy data
def make_a_model(pairNo, x, X, y, dy, theta0=1e-3, thetaL=1e-3, thetaU=1):
    gp = GaussianProcess(corr='squared_exponential', # If the original experiment is known to be infinitely differentiable (smooth), then one should use the squared-exponential correlation model.
                         regr = "quadratic", #?
                         theta0 = theta0,
                         thetaL = thetaL,
                         thetaU = thetaU,
                         nugget = (dy / y) ** 2, #?
                         random_start=500)
# Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, MSE = gp.predict(x, eval_MSE=True)
    sigma = np.sqrt(MSE)
    return y_pred, sigma

# Make a GaussianProcess model for data without noise (It complains for TDC dataset though!)
def make_a_perfect_model(pairNo, x, X, y):
    gp = GaussianProcess(theta0=1e-3,
                         thetaL=1e-3,
                         thetaU=1,
                         random_start=500)
# Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, MSE = gp.predict(x, eval_MSE=True)
    sigma = np.sqrt(MSE)
    return y_pred, sigma

# Plot data points with error bars
def plot_data(ax, pairNo, dt, X, y, dy, ob):
# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
    ax.errorbar(X.ravel() + dt, y, dy, fmt='.c', markersize=10, label='Observations ' + ob)
    return ax

# Plot the best-fit model of data along with 95% uncertainties
def plot_model(ax, pairNo, dt, x, y_pred, sigma, ob):
# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
    ax.plot(x + dt, y_pred, '-', c='k', linewidth=2, label='Prediction ' + ob)
    ax.fill(np.concatenate([x + dt, x[::-1] + dt]),
            np.concatenate([y_pred - 1.9600 * sigma,
                            (y_pred + 1.9600 * sigma)[::-1]]),
            alpha=.5, color='darkorange', ec='None', 
            label='95% confidence interval ' + ob)
    return ax

# Cross correlate two arrays (models or data!) and return the index of the maximum correlation and the full corr array
def cross_correlate_models(model1, model2, mode='same'):
    corr = correlate(model1, model2, mode=mode)
    abscorr = np.abs(corr)
    maxcorr = np.max(abscorr)
    return np.where(abscorr == maxcorr)[0], corr


# Fit GP model to each sampling window of the data separately
N_eval = 25000
path = "/home/saas9842/PhD/Courses/AstroML/Project/tdc1/rung3/"
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
    XA = XA.reshape((len(XA), 1))
    yA = (lightcurve['lcA'] - np.mean(lightcurve['lcA'])) / np.std(lightcurve['lcA'])
    dyA = (lightcurve['errA'] - np.mean(lightcurve['errA'])) / np.std(lightcurve['errA'])
    y_predA, sigmaA = make_a_model(pairNo, x, XA, yA, dyA, theta0=sig, thetaL=tau, thetaU=tau) #A
    
    XB = (lightcurve['time']).T
    XB = XB.reshape((len(XB), 1))
    yB = (lightcurve['lcB'] - np.mean(lightcurve['lcB'])) / np.std(lightcurve['lcB'])
    dyB = (lightcurve['errB'] - np.mean(lightcurve['errB'])) / np.std(lightcurve['errB'])
    y_predB, sigmaB = make_a_model(pairNo, x, XB, yB, dyB, theta0=sig, thetaL=tau, thetaU=tau) #B
    

# Cross correlate the two models
    ind_maxcorr, corr = cross_correlate_models(y_predA, y_predB)
    maxcorr = x[ind_maxcorr]
#    t_maxcorr = t_maxcorr[0][0]

    if corr[ind_maxcorr] < 0:
        t_maxcorr = -maxcorr[0][0]
    else:
        t_maxcorr = maxcorr[0][0]



    print "Pair " + str(pairNo) + " done!"
    
    # Plot everything
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    axs = axs.flatten()
    axA = axs[0]
    axB = axs[1]

    axA = plot_data(axA, pairNo, 0, XA, yA, dyA, "A") # A
    axA = plot_model(axA, pairNo, 0, x, y_predA, sigmaA,"A ") # A
    
    axB = plot_data(axB, pairNo, 0, XB, yB, dyB, "B") # B
    axB = plot_model(axB, pairNo, 0, x, y_predB, sigmaB,"B ") # B
    
    axA.set_ylim(-4, 3)
    axB.set_ylim(-4, 3)
    plt.xlabel('t [days]')
    plt.ylabel('normalized flux [arbitrary]')
    fig.suptitle('PairNo: ' + str(pairNo) +
              ' - true time delay: ' + str(dt_true) +
              ' - maximum correlation at: ' + str(t_maxcorr))
    
    plt.show()

'''

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
    

    figA.suptitle("Lightcurve A \n Estimated time delay: " + 
                  str(np.mean(t_maxcorr)) + 
                  " days")
    figB.suptitle("Lightcurve B \n True time delay: " + 
                  str(dt_true) + 
                  " days")
    figA.savefig("pair" + str(pairNo) + "_A.png")
    figB.savefig("pair" + str(pairNo) + "_B.png")
#plt.show()
'''

