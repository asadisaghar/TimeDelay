import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from sklearn.gaussian_process import GaussianProcess
import numpy as np
import cPickle as pickle
from matplotlib.mlab import frange

def make_a_model(pairNo, x, X, y, dy, theta0=1e-3, thetaL=1e-3, thetaU=1):
    """Make a GaussianProcess model for noisy data"""

    # If the original experiment is known to be infinitely
    # differentiable (smooth), then one should use the
    # squared-exponential correlation model.
    gp = GaussianProcess(corr='squared_exponential',
                         regr = "quadratic",
                         theta0 = theta0,
                         thetaL = thetaL,
                         thetaU = thetaU,
                         nugget = (dy / y) ** 2,
                         random_start=500)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, MSE = gp.predict(x, eval_MSE=True)
    sigma = np.sqrt(MSE)
    return gp, y_pred, sigma

def make_a_perfect_model(pairNo, x, X, y):
    """Make a GaussianProcess model for data without noise (It
    complains for TDC dataset though!)"""
    gp = GaussianProcess(theta0=1e-3,
                         thetaL=1e-3,
                         thetaU=1,
                         random_start=500)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, MSE = gp.predict(x, eval_MSE=True)
    sigma = np.sqrt(MSE)
    return gp, y_pred, sigma


def eval_signal_from_GP_model(window, t_eval):
    with open("GPModels/%s.pkl" % window, "rd") as f:
        model = pickle.load(f)
    sig_eval, MSE = model.predict(t_eval, eval_MSE=True)
    return sig_eval, MSE

def evenly_sample_window(window_data, dt):
    ts = frange(window_data[0]['time'], window_data[-1]['time'], dt)
    return ts.reshape((len(ts), 1))

def resample_using_gp_models(data, pair_ids, dt=0.1):
    outdata = None
    for pair_id in pair_ids:
        print "Resampling pair %s" % pair_id
        pair_data = data[data['full_pair_id'] == pair_id]
        tau = pair_data[0]['tau']
        sig = pair_data[0]['sig']
        dt_true = pair_data[0]['dt']
        m1 = pair_data[0]['m1']
        m2 = pair_data[0]['m2']

        windows = np.unique(pair_data['window_id'])
        for i, window in enumerate(windows):
            print "    window %s" % window
            window_data = pair_data[pair_data['window_id'] == window]
            t_eval = evenly_sample_window(window_data, dt)

            sig_evalA, sig_errA = eval_signal_from_GP_model(str(window) + "A", t_eval)
            sig_evalA = np.reshape(sig_evalA, (len(sig_evalA), 1))
            sig_errA = np.reshape(sig_errA, (len(sig_errA), 1))
            sig_evalB, sig_errB = eval_signal_from_GP_model(str(window) + "B", t_eval)
            sig_evalB = np.reshape(sig_evalB, (len(sig_evalB), 1))
            sig_errB = np.reshape(sig_errB, (len(sig_errB), 1))

            res = np.zeros((len(t_eval), 1), dtype=[('window_id', 'f4'), 
                                                    ('t_eval', 'f4'),
                                                    ('sig_evalA', 'f4'),
                                                    ('sig_errA', 'f4'),
                                                    ('sig_evalB', 'f4'),
                                                    ('sig_errB', 'f4'),
                                                    ('dt', 'f4'),
                                                    ('tau', 'f4'),
                                                    ('sig', 'f4'),
                                                    ('m1', 'f4'),
                                                    ('m2', 'f4')])
            
            res['window_id'] = window
            res['t_eval'] = t_eval
            res['sig_evalA'] = sig_evalA
            res['sig_errA'] = sig_errA
            res['sig_evalB'] = sig_evalB
            res['sig_errB'] = sig_errB
            res['dt'] = dt_true
            res['tau'] = tau
            res['sig'] = sig
            res['m1'] = m1
            res['m2'] = m2

            if outdata is None:
                outdata = res
            else:
                outdata = np.append(outdata, res)

    return outdata
