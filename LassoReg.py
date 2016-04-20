# Perform a Lasso regression on the entire dataset
# Features: evenly-sampled lightcurves of images A and B in each sampling window
# labels: dt_true, tau, sigma[, m1, m2]

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.linear_model import Lasso, LassoCV
from sklearn.cross_validation import train_test_split

# Read fetures

# Read lables

# Split data to train and test sets (cross validation is done inside LassoCV, sob no need for a CV set)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, 
                                                                            test_size=0.33, #How to make sure it does not split the data in the middle of a window or pair? 
                                                                            random_state=42)

# Guess a range of alphas (regularization parameter)
alphas = 10**np.linspace(-5, 1, 100)

# Build a LassoCV model
# Train a Lasso model over alphas and choose the best alpha by cross-validation
model = LassoCV(alphas=alphas, selection='random', intercept=True, normalize=True) 
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
