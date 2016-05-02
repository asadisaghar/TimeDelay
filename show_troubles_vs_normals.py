
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt


# In[3]:

data = np.load("TimeDelayData/dt_correlate_normalized_wgtd.npz")['arr_0']


# In[ ]:

data.shape


# In[4]:

troubled_data = data[data['est_dt_median'] == 0]
normal_data = np.setdiff1d(data, troubled_data)


# In[5]:

plt.errorbar(normal_data['dt'], -normal_data['est_dt_median'], yerr=np.sqrt(2. * normal_data['est_dt_std']), fmt='.c', alpha=0.5)
plt.plot(normal_data['dt'], -normal_data['est_dt_median'], '.b', label='normal data : %s windows'%(len(normal_data)))
plt.plot(normal_data['dt'], normal_data['dt'], '-r', alpha=0.8)
#plt.plot(normal_data['dt'], -normal_data['dt'], '-r', alpha=0.8)
yminmax = np.min(np.abs(data['dt']))
plt.axhspan(ymin=-yminmax, ymax=yminmax, facecolor='r', alpha=0.5)
plt.xlabel('dt_true (days)')
plt.ylabel('dt_sindow median (days)')
plt.plot(troubled_data['dt'], -troubled_data['est_dt_median'], 'ok', lw=0, alpha=0.9, label='troubled_data: %s windows'%(len(troubled_data)))
plt.legend().draggable()
plt.show()


# In[ ]:

mod_data = np.zeros((len(normal_data),))
for i in range(len(normal_data)):
    if normal_data['est_dt_median'][i]<0:
        mod_data[i] = normal_data['est_dt_median'][i]-normal_data['est_dt_std'][i]
    else:
        mod_data[i] = normal_data['est_dt_median'][i]+normal_data['est_dt_std'][i]

plt.plot(np.abs(normal_data['dt']), np.abs(mod_data), 'ok', lw=0, alpha=0.3)
plt.show()

