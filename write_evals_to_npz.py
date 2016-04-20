import numpy as np
import cPickle as pickle
from matplotlib.mlab import frange

def eval_signal_from_GP_model(window, t_eval):
    with open("GPModels/%s.pkl" % window, "rd") as f:
        model = pickle.load(f)
    sig_eval, MSE = model.predict(t_eval, eval_MSE=True)
    return sig_eval, MSE

def evenly_sample_window(window_data, dt):
    return frange(window_data[0], window_data[-1], dt)

def filter_pairs(pair_ids, system_type="double"):
    quad_pair_ids = pair_ids[pair_ids % 1.0 == 0.5] # Finde file A of a quadratic system
    quad_pair_ids = np.append(quad_pair_ids, quad_pair_ids - 0.5) # Finds file B of the same quadratic system
    if system_type == "double":
        return np.setdiff1d(pair_ids, quad_pair_ids) # Removes all quadratic systems from pair_ids
    elif system_type == "quad":
        return quad_pair_ids # Chooses all quadratic systems from pair_ids
    return  pair_ids

def resample_using_gp_models(data, pair_ids, dt=0.1):
    outdata = None
    for pair_id in pair_ids:
        pair_data = data[data['full_pair_id'] == pair_id]
        tau = pair_data[0]['tau']
        sig = pair_data[0]['sig']
        dt_true = pair_data[0]['dt']
        m1 = pair_data[0]['m1']
        m2 = pair_data[0]['m2']

        windows = np.unique(pair_data['window_id'])
        for i, window in enumerate(windows):
            window_data = pair_data[pair_data['window_id'] == window]
            t_eval = evenly_sample_window(window_data, dt)
            sig_evalA, sig_errA = eval_signal_from_GP_model(str(window) + "A", t_eval)
            sig_evalB, sig_errB = eval_signal_from_GP_model(str(window) + "B", t_eval)

            res = zeros(len(t_eval), dtype=(('window_id', 'f4'), 
                                            ('sig_evalA', 'f4'),
                                            ('sig_errA', 'f4'),
                                            ('sig_evalB', 'f4'),
                                            ('sig_errB', 'f4'),
                                            ('dt', 'f4'),
                                            ('tau', 'f4'),
                                            ('sig', 'f4'),
                                            ('m1', 'f4'),
                                            ('m2', 'f4')))

            res['window_id'] = window_id
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


def write_to_file(dt=0.1, system_type="double"):
    data = np.load("TimeDelayData/pairs_with_truths_and_windows.npz")['arr_0']
    pair_ids = np.unique(data['full_pair_id'])
    pair_ids = filter_pairs(pair_ids, system_type)
    outdata = resample_using_gp_models(data, pair_ids)
    np.savez('GPModels_of_windows_with_truth', outdata)
