import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics 
import sklearn
import seaborn as sns

def testing_data(data_set, params, plot = 'y', save=False):
    os.chdir(data_set.ip['outdir'])
    """
        Function which evaluates the MSE/MAE of the test setr
        Inputs:
         data_seta : data container
         params    : new params evaluated through the lrrde procedure
         plot      : flag if 'y' plot results
        Returns:
            evaluation of MAE and  MSE of the prediction on the training set
    """            
    if plot == 'y':
        y_ref  = data_set.y_test
        y_test = np.dot(data_set.x_test, params.T).reshape(np.shape(y_ref)).T
        y_test = y_test*data_set.ip['sigma']
        y_test_s     = np.sort(y_test,axis=0)
        y_ref_s      = np.sort(y_ref, axis=0)

        mse_test     = np.mean(np.power(y_test_s - y_ref_s, 2))
        mae_test_s   = np.mean(np.abs(y_ref_s    - y_test_s.T))

        print('MSE (lrr-de) = {}'.format(mse_test))
        print('MAE (lrr-de) = {}'.format(mae_test_s))
        print("-------")
            
        plt.figure(dpi=100)    
        plt.title("Comparison prediction Test Set")
        plt.plot(y_test,label="lrr-de",marker="*")
        plt.plot(y_ref, label="ref",marker="*")
        plt.legend(loc='upper left')
        if save:
            plt.savefig("TestSet.pdf")


