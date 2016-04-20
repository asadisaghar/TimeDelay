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
