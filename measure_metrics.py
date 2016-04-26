import numpy as np

def measure_metrics(dataset, measurementset):
    fulldata = np.load("TimeDelayData/%s"%(dataset))['arr_0']
    N_data = np.unique(fulldata['pair_id']).shape
    data = np.load("TimeDelayMeasurements/%s"%(measurementset))['arr_0']
    N_measured = np.unique(data['pair_id']).shape
    
    dt_trues = data['dt']
    dt_ests = data['est']
    dt_est_errs = data['est_err']

    # success fraction
    f =  N_measured / N_data

    # Chi2 value
    normalization = 1. / (f * N)
    diff = ((dt_ests - dt_trues) / dt_est_err) ** 2
    chi2 = normalization * diff.sum()

    # precision
    prec = dt_est_err / dt_trues
    P = normalization * prec.sum()

    # accuracy or bias
    bias = ((dt_ests - dt_trues) / dt_trues)
    A = normalization * bias.sum()

    return f, chi2, P, A

if __name__ == '__main__':
    call_some_function(*sys.argv[1:])
