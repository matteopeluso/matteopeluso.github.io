#from tool import *
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def fast_loocv_MO(data_set, new_values):
    """
        This function takes as input a data_set and a set of new values
        and evaluates the Leave One Out Cross Validation Error as
        the equation reported on the article

        Inputs:
            data set   : data container
            new values : np array on new coefficient on which evaluate the fast loocv
        Returns:
            loocv error, mean absolute value
    """
    y    = data_set.y_sample
    v    = data_set.w_vector
    
    sigma, H = sigma_h(data_set)
    
    l  = leverage(data_set,flag="vs")
    
    err_loocv = []
    mae       = []
    for c in new_values:
        _err_loocv, _mae = fast_loocv(c, H, y, l, v)
        err_loocv.append(np.asscalar(_err_loocv))
        mae.append(_mae)
    return err_loocv, mae


def fast_loocv(c, H, y, l, v):
    """
        Mathematical evaluation of the loocv
    """
    y_sample_est = np.dot(H,c)
    dev          = np.subtract(y_sample_est, y)
    den          = 1 - l
    loocv_ei     = np.power(np.divide(dev.T, den), 2)
    aux          = np.dot(loocv_ei.T, v)    
    mae          = np.mean(np.abs(dev)) 
    return aux, mae
