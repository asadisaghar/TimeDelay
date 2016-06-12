import sys
import numpy as np

argv = sys.argv[1:]

if not argv:
    print """Usage: split_according_to_prediction.py START END PREDICTIONS.npz INPUT_DATASET.npz OUTPUT_DATASET.npz
START and END are in % of all pairs, sorted by prediction error (ascending order).

Example:

python split_according_to_prediction.py \
   99 100 \
   TimeDelayData/timeshift_correlate_normalized_detrend.measures.pairsums.ests.npz \
   TimeDelayData/gp_resampled_of_windows_with_truth.npz \
   TimeDelayData/gp_resampled_of_windows_with_truth.predictivity:99-100.npz
"""
    sys.exit(1)

start, end, preds, input, output = argv

start = float(start) / 100.
end = float(end) / 100.

preds = np.load(preds)['arr_0']
preds = preds[np.argsort(np.abs(preds['dt'] - preds['est']))]

start = np.floor(start * len(preds))
end = np.floor(end * len(preds))


data = np.load(input)['arr_0']

pair_ids = preds['pair_id'][start:end]

data = data[np.in1d(data['pair_id'], pair_ids)]

np.savez(output, data)
