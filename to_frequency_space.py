import numpy as np


data = np.load("TimeDelayData/gp_resampled_of_windows_with_truth.npz")['arr_0']

window_ids = np.unique(data['window_id'])

fft_out = None
for window_id in window_ids:
    window_data = data[data['window_id'] == window_id]

    freq = np.fft.fftfreq(len(window_data))

    res = np.zeros(len(window_data), dtype=[('window_id', 'f4'), ('fftA', 'f4'), ('fftB', 'f4'), ('dt', '<f4'), ('tau', '<f4'), ('sig', '<f4'), ('m1', '<f4'), ('m2', '<f4')])
    res['fftA'] = np.fft.fft(window_data['sig_evalA'])
    res['fftB'] = np.fft.fft(window_data['sig_evalB'])
    for name in ('dt', 'tau', 'sig', 'm1', 'm2'):
        res[name] = window_data[name]

    if fft_out is None:
        fft_out = res
    else:
        fft_out = np.append(fft_out, res)

np.savez("TimeDelayData/fft_of_windows_with_truth.npz", fft_out)
