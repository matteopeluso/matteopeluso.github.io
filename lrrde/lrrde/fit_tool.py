import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score

def standardize_data(data):
    """
        Function which takes as input a data array and returns his mean, standard deviation and the data divided by his std

        Inputs:
            data : np array to be standardize

        Returns:
            mean value, standard deviation and standardize data
    """
    mean     = np.mean(data)
    sigma    = np.std(data)
    data_std = np.divide(data, sigma) 
    return mean, sigma, data_std

def leverage(data_set):
    """
        Function which leverage data

        Inputs:
            dataset : data container

        Returns:
            leverage of the input data
    """
    # QM Data
    y    = data_set.y_train
    H    = data_set.x_train
    v    = np.ones(np.shape(y))/np.std(y)
    
    n    = np.size(y,0)
    V    = np.tile(v,(np.size(H,1),1))
    
    H = np.dot(H,V)
    
    mean = np.mean(H)
    dev  = H - np.ones(np.shape(H))
    num  = np.sum(np.power(dev,2),1)
    den  = np.sum(num)
    
    l = 1/n + num/den
    return l


def eval_r2_mse(data_set, params):
    """
        Evaluate the R2/MSE/MAE of the output
        and compares it with the old values
        
        Inputs:
          data_set : data container
          params   : new params
        Returns:
            MSE, MAE, R2 on the Prediction evaluation on the Training Set
    """
    y_ref    = data_set.y_train
    Ht       = data_set.x_train
                    
    # Eval metrics lrr-de
    y_test    = np.dot(Ht, params.T).reshape(np.shape(y_ref)).T
    y_test    = y_test * data_set.ip['sigma']
    r2_lrrde  =  r2_score(y_ref, y_test)
    mse_lrrde = np.mean(np.power(y_test - y_ref, 2))
    mae_lrrde = np.mean(np.abs(y_test - y_ref.T))
    print("R2 score:  lrr-de", r2_lrrde)
    print("MSE score: lrr-de", mse_lrrde)
    print("MAE score: lrr-de", mae_lrrde)

    
    fig = plt.figure(dpi = 100)
    sns.set_context("paper")
    plt.plot(y_ref, "r-", label="ref" )
    plt.plot(y_test,"b-", label="lrr-de" )
    plt.legend(loc='upper left')
    if save:
            plt.savefig("PredictionTrainingSet.pdf")
        
    return r2_lrrde, mse_lrrde


