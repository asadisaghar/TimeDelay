import numpy as np

freq_count = 100

data = np.load("TimeDelayData/fft_of_windows_with_truth.npz")['arr_0']
window_ids = np.unique(data['window_id'])

copy_cols = ['window_id', 'dt', 'tau', 'sig', 'm1', 'm2']

res = np.zeros(len(window_ids), dtype=([('window_id', 'f4'), ('dt', '<f4'), ('tau', '<f4'), ('sig', '<f4'), ('m1', '<f4'), ('m2', '<f4')] +
                                       [('fft_diff_angle_%s' % i, 'f4') for i in xrange(0, freq_count)] +
                                       [('fft_diff_abs_%s' % i, 'f4') for i in xrange(0, freq_count)] +
                                       [('fft_diff_freq_%s' % i, 'f4') for i in xrange(0, freq_count)]))

for idx, window_id in enumerate(window_ids):
    window_data = data[data['window_id'] == window_id]
    fftA = window_data['fftA']
    fftB = window_data['fftB']
    freq = np.fft.fftfreq(len(window_data))

    order = np.argsort(-np.abs(fftA)/np.max(np.abs(fftA)) - np.abs(fftB)/np.max(np.abs(fftB)))[:freq_count]

    diff = fftA[order] - fftB[order]
    fft_diff_angle = np.angle(diff)
    fft_diff_abs = np.abs(diff)
    fft_diff_freq = freq[order]

    for col in copy_cols:
        res[idx][col] = window_data[0][col]

    for freq_idx in xrange(0, freq_count):
        res[idx]['fft_diff_angle_%s' % freq_idx] = fft_diff_angle[freq_idx]
        res[idx]['fft_diff_abs_%s' % freq_idx] = fft_diff_abs[freq_idx]
        res[idx]['fft_diff_freq_%s' % freq_idx] = fft_diff_freq[freq_idx]

np.savez("TimeDelayData/fft_comparisons_with_truth.npz", res)
