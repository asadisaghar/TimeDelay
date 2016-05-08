# python regression.py est_dt_mean,est_dt_median,est_dt_std 2 dt \
#   TimeDelayData/timeshift_correlate_normalized_detrend.measures.pairsums.clusters.npz \
#   TimeDelayData/timeshift_correlate_normalized.measures.pairsums.clusters.npz \
#   TimeDelayData/timeshift_mse_normalized_detrend.measures.pairsums.clusters.npz \
#   TimeDelayData/timeshift_mse_normalized.measures.pairsums.clusters.npz \
#   model.pkl 

# python predict.py est_dt_mean,est_dt_median,est_dt_std 2 dt \
#   TimeDelayData/timeshift_correlate_normalized_detrend.measures.pairsums.npz \
#   TimeDelayData/timeshift_correlate_normalized.measures.pairsums.npz \
#   TimeDelayData/timeshift_mse_normalized_detrend.measures.pairsums.npz \
#   TimeDelayData/timeshift_mse_normalized.measures.pairsums.npz \
#   model.pkl 


python regression.py est_dt_mean,est_dt_median,est_dt_std 2 dt \
  TimeDelayData/timeshift_correlate_normalized_detrend.measures.pairsums.clusters.npz \
  model.pkl 

python predict.py est_dt_mean,est_dt_median,est_dt_std 2 dt \
  TimeDelayData/timeshift_correlate_normalized_detrend.measures.pairsums.npz \
  model.pkl 
