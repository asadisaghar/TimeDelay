# Perform a Lasso regression on the entire dataset
# Features: evenly-sampled lightcurves of images A and B in each sampling window
# labels: dt_true, tau, sigma[, m1, m2]

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.linear_model import Lasso, LassoCV
from sklearn.cross_validation import train_test_split

data = np.load("TimeDelayData/gp_resampled_of_windows_with_truth.npz")['arr_0']

feature_names = ['sig_evalA', 'sig_errA', 'sig_evalB', 'sig_errB']
label_names = ['dt', 'tau', 'sig', 'm1', 'm2']
feature_formats = ['float64'] * len(feature_names)
label_formats = ['float64'] * len(label_names)

# # Read fetures
# features = np.zeros(len(data), dtype=[('sig_evalA', np.float64), 
#                                       ('sig_errA', np.float64),
#                                       ('sig_evalB', np.float64),
#                                       ('sig_errB', np.float64)])

# # Read lables
# labels = np.zeros(len(data), dtype=[('dt', np.float64),
#                                     ('tau', np.float64),
#                                     ('sig', np.float64),
#                                     ('m1', np.float64),
#                                     ('m2', np.float64)])

# Read fetures
features = np.zeros(len(data), dtype={'names': feature_names, 
                                       'formats': feature_formats})

# Read lables
labels = np.zeros(len(data), dtype={'names': label_names, 
                                       'formats': label_formats})


for name in feature_names:
    features[name] = data[name]

for name in label_names:
    labels[name] = data[name]


# Split data to train and test sets by window_id
# Cross validation is done inside LassoCV, so no need for a separate CV set

windows = np.unique(data['window_id'])
test_size = int(len(windows) * 0.33)
test_windows = windows[:test_size]
train_windows = windows[test_size+1:]
test_end_ind = np.where(data['window_id'] == test_windows[-1])[0][-1]
features_test = features[:test_end_ind]
features_test = np.reshape(features_test, (len(features_test), 1))
labels_test = labels[:test_end_ind]
features_train = features[test_end_ind + 1:]
features_train = np.reshape(features_train, (len(features_train), 1))
labels_train = labels[test_end_ind + 1:]

# Guess a range of alphas (regularization parameter)
alphas = 10**np.linspace(-5, 1, 100)

# Build a LassoCV model
# Train a Lasso model over alphas and choose the best alpha by cross-validation
model = LassoCV(alphas=alphas, selection='random', fit_intercept=True, normalize=True) 
model.fit(features_train, labels_train)
w = model.coef_

# Plot feature contributions for the best-fit model asd the MSE for each alpha
fig, axs = plt.subplots(1,3)
axs = axs.flatten()

axs[0].plot(model.alphas_, model.mse_path_)
axs[1].plot(range(len(w)), w, 'o')

# How well does the trained model perform on the test test?
labels_pred = model.predict(features_test)
score = np.mean((labels_pred - labels_test) ** 2)
axs[2].plot(labels_pred, labels_test)
