# TimeDelay
Final project for the AstroML course, based on data from TDC1
Page of the challeng, including data:
http://timedelaychallenge.org/


# Data format

Columns in 'x = load("pairs_with_truths_and_windows.npz")['arr_0']':


  * From the original light curve pair files: 
    * time - 
    * lcA - 
    * errA - 
    * lcB - 
    * errB - 

  * From the path of the light curve pair files
    * tdc - numerical part of tdc directory name
    * rung - numerical part of rung directory name
    * pair - numerical part of pair filename. Note: pairs in quad
      systems (A, B) have 0.0 added for A and 0.5 added for B.

  * From the corresponding truth file (duplicated for all rows in the corresponding light curve)
    * dt - 
    * m1 - 
    * m2 - 
    * zl - 
    * zs - 
    * id - 
    * tau - 
    * sig - 

  * Computed values
    * full_pair_id - a (globally) unique id for each combination of tdc, rung, pair
    * window_id - a (globally) unique id for each sample window in the light curves


# Scripts

## convert_to_npz.py
Input: tdc0/*, tdc1/*
Output: TimeDelayData/pairs_with_truths.npz

## amend_sampling.py
Input: TimeDelayData/pairs_with_truths.npz
Output: TimeDelayData/pairs_with_truths_and_windows.npz

## GP.py
Input: TimeDelayData/pairs_with_truths_and_windows.npz
Output: GPModels/*

## gp_resample.py
Input: TimeDelayData/pairs_with_truths_and_windows.npz, GPModels/*
Output: TimeDelayData/gp_resampled_of_windows_with_truth.npz

## Timeshift.py

Input: TimeDelayData/gp_resampled_of_windows_with_truth.npz
Output: timeshift_{correlate,ng_correlate,mse}_normalized[_detrend].npz

## measure_window_corelations.py
Input: timeshift_{correlate,ng_correlate,mse}_normalized[_detrend].npz
Output: timeshift_{correlate,ng_correlate,mse}_normalized[_detrend].measures.npz

## sum_pair_corelation_measures.py
Input: timeshift_{correlate,ng_correlate,mse}_normalized[_detrend].measures.npz
Output: timeshift_{correlate,ng_correlate,mse}_normalized[_detrend].measures.pairsums.npz


