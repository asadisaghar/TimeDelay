# Perform a Lasso regression on the entire dataset
# Features: evenly-sampled lightcurves of images A and B in each sampling window
# labels: dt_true, tau, sigma[, m1, m2]

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.linear_model import Lasso, LassoCV
from sklearn.cross_validation import train_test_split
from timedelay.data_splitting import *

data = np.load("TimeDelayData/gp_resampled_of_windows_with_truth.npz")['arr_0']
print "data shape: ", data.shape
feature_names = ['sig_evalA', 'sig_errA', 'sig_evalB', 'sig_errB']
label_names = ['dt', 'tau', 'sig', 'm1', 'm2']
feature_formats = ['float64'] * len(feature_names)
label_formats = ['float64'] * len(label_names)

# Read fetures
#features = np.zeros((len(data), len(feature_names)))
features = np.zeros((len(data), 1))

# Read lables
labels = np.zeros((len(data), len(label_names)))

#for i, name in enumerate(feature_names):
#    features[:, i] = data[name]


features[:, 0] = ((data['sig_evalA']-np.mean(data['sig_evalA'])) - (data['sig_evalB'] - np.mean(data['sig_evalB'])))
print "features shape: ", features.shape

for j, name in enumerate(label_names):
    labels[:, j] = data[name]
print "labels shape: ", labels.shape

features_train, labels_train, features_test, labels_test = split_data_by_windows(data, features, labels, test_fraction=0.33)

print "features_train shape: ", features_train.shape
print "features_test shape: ", features_test.shape
print "labels_train shape: ", labels_train.shape
print "labels_test shape: ", labels_test.shape

# Guess a range of alphas (regularization parameter)
alphas = 10 ** np.linspace(-5, 1, 100)

# Build a LassoCV model
# Train a Lasso model over alphas and choose the best alpha by cross-validation
model = LassoCV(alphas=alphas, selection='random', fit_intercept=True, normalize=True, n_jobs=4) 
model.fit(features_train, labels_train[:,0])
w = model.coef_
print w

# Pickle the model
with open("LassoReg.pkl", "wd") as f:
    pickle.dump(model, f)

# Plot feature contributions for the best-fit model asd the MSE for each alpha
fig, axs = plt.subplots(1,3)
axs = axs.flatten()

axs[0].plot(model.alphas_, model.mse_path_)
axs[1].plot(range(len(w)), w, 'o')

# How well does the trained model perform on the test test?
labels_pred = model.predict(features_test)
score = np.mean((labels_pred - labels_test[:, 0]) ** 2)
axs[2].plot(labels_pred, labels_test[:, 0], '.')
axs[2].plot(labels_test[:, 0], labels_test[:, 0], '-k') 

plt.show()
