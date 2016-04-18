import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from sklearn.gaussian_process import GaussianProcess
do_plot = False
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
N_eval = 5000
path = "/home/saas9842/PhD/Courses/AstroML/Project/tdc1/rung3/"
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
#plt.show()


