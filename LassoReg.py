# Perform a Lasso regression on the entire dataset
# Features: evenly-sampled lightcurves of images A and B in each sampling window
# labels: dt_true, tau, sigma[, m1, m2]

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.linear_model import Lasso, LassoCV, MultiTaskLassoCV
from sklearn.cross_validation import train_test_split
from timedelay.data_splitting import *

x = np.load("TimeDelayData/gp_resampled_of_windows_with_truth.npz")['arr_0']
#x = np.load("TimeDelayData/fft_comparisons_with_truth.npz")['arr_0']
np.random.shuffle(x)

xcols = ['sig_evalA', 'sig_errA', 'sig_evalB', 'sig_errB'] # Features
#xcols = ['sig_evalA', 'sig_evalB']
ycols = ['dt', 'tau', 'sig', 'm1', 'm2'] #labels


X = np.zeros((len(x), len(xcols)))
for i, xcol in enumerate(xcols):
    X[:,i] = x[xcol]

y = np.zeros((len(x), len(ycols)))
for j, ycol in enumerate(ycols):
    y[:,j] = x[ycol]

num_rows = len(X)
train_len = int(num_rows * 3. / 4.)
Xtrain = X[:train_len,:]
ytrain = y[:train_len]
Xtest = X[train_len:,:]
ytest = y[train_len:]

######################
## MultiTaskLassoCV ##
######################
# # Guess a range of alphas (regularization parameter)
# alphas = 10 ** np.linspace(-5, -1, 100)

# # Build a LassoCV model
# # Train a Lasso model over alphas and choose the best alpha by cross-validation
# model = MultiTaskLassoCV(alphas=alphas, selection='random', cv=3, fit_intercept=True, normalize=False, n_jobs=4) 
# model.fit(Xtrain, ytrain)
# ypred = model.predict(Xtest)
# w = model.coef_
# print w

# score = (sum((ytest - ypred)**2),)

# # Plot feature contributions for the best-fit model asd the MSE for each alpha
# #axs[0].plot(ytest, ypred, '.')
# #axs[0].set_xlim(np.min(ytest[:,0]), np.max(ytest[:,0]))
# #axs[0].set_ylim(np.min(ytest[:,0]), np.max(ytest[:,0]))
# axs[0].plot(np.log(model.alphas_), np.sqrt(model.mse_path_).mean(axis = 1))
# axs[0].axvline(-np.log(model.alpha_), color = 'red')
# axs[0].set_ylabel('RMSE (avg. across folds)')
# axs[0].set_xlabel('log(alpha)')
# axs[1].plot(range(len(w)), w, 'o')
# axs[1].hlines(0, xmin=0, xmax=len(w), colors='k')
# plt.show()

###########
## Lasso ##
###########
# Guess a range of alphas (regularization parameter)
alphas = 10 ** np.linspace(-5, -1, 10)

fig, axs = plt.subplots(2,2)
axs = axs.flatten()

for alpha in alphas:
# Build a LassoCV model
# Train a Lasso model over alphas and choose the best alpha by cross-validation
    model = Lasso(alpha=alpha, selection='random', fit_intercept=True, normalize=False) 
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    w = model.coef_
    score = (sum((ytest - ypred)**2),)
    axs[0].plot(ytest, ypred, '.')
    axs[1].plot(np.log(alpha), score, '.')
    axs[2].plot(range(len(w)), w, 'o')
# Plot feature contributions for the best-fit model asd the MSE for each alpha
#axs[0].plot(ytest, ypred, '.')
#axs[0].set_xlim(np.min(ytest[:,0]), np.max(ytest[:,0]))
#axs[0].set_ylim(np.min(ytest[:,0]), np.max(ytest[:,0]))
plt.show()












# Pickle the model
#with open("LassoReg.pkl", "wd") as f:
#    pickle.dump(model, f)


