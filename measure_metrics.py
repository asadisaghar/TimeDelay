import numpy as np

def measure_metrics(dataset, measurementset):
    true_val = np.load("TimeDelayData/%s"%(dataset))['arr_0']
    N_data = np.unique(true_val['pair_id']).shape
    est_val = np.load("TimeDelayMeasurements/%s"%(measurementset))['arr_0']
    N_measured = np.unique(est_val['pair_id']).shape
    
    # success fraction
    f =  N_measured / N_data

    # Chi2 value
    normalization = 1. / (f * N)
    diff = ((est_val['dt'] - true_val['dt']) / est_val['errdt']) ** 2
    chi2 = normalization * diff.sum()

    # precision
    precision = est_val['errdt'] / true_val['dt']
    P = normalization * precision.sum()

    # accuracy or bias
    bias = ((est_val['dt'] - true_val['dt']) / true_val['dt'])
    A = normalization * bias.sum()

    return f, chi2, P, A

if __name__ == '__main__':
    call_some_function(*sys.argv[1:])
