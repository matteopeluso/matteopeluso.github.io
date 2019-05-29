import os
import numpy as np
#from tool import *
import matplotlib.pyplot as plt

from sklearn import metrics 
import sklearn
import seaborn as sns

def testing_data(data_set, params, plot = 'y'):
    """
        Function which evaluates the MSE/MAE of the test setr
        
        Inputs:
             data_set : data container
             params   : new params
             old_params : old params
             plot       :  flag if 'y' plot results
    """
    os.chdir(data_set.input_params['outdir'])

#     if data_set.input_params['n_functions'] in [0,1,2]:
#         params = np.append([data_set.input_params['q_value']], params)
            
    if plot == 'y':
        sns.set_context("paper")
        #fig, axs = plt.subplots(4,1, figsize=(10, 20), facecolor='w', edgecolor='k')
        
        count_plot = 0
    if data_set.input_params['n_functions'] in [0,1,2]:
        aux = [data_set.input_params['q_value']]
        aux.extend(params.tolist()[0])
        params = np.array(aux)
                
    for i in range(2):
        if i == 0:
            data = data_set.energy_ts
        elif i == 1:
            data = data_set.force_ts
            
        for _n_water in data_set.input_params['N_water']:
            print("SET: {}, # of water molecules {}".format(data.flag, _n_water))
            y_ref  = data.training_set['y_test{}'.format(_n_water)]

            q   = data.test_set['q_test{}'.format(_n_water)]
            c6  = data.test_set['c6_test{}'.format(_n_water)]
            c12 = data.test_set['c12_test{}'.format(_n_water)]             

            
            if data_set.input_params['n_vs'] == 0:
                if data_set.input_params['n_functions'] == 3 :
                    Hvs = np.matrix([q, c12, c6]).T   # MxN functionts
                elif data_set.input_params['n_functions'] == 0 :
                    Hvs = np.matrix([q, c12, c6]).T
                    
            elif data_set.input_params['n_vs'] == 2:
                # VS Descriptor
                q_vs1   = data.test_set['q_test_vs1{}'.format(_n_water)]
                c6_vs1  = data.test_set['c6_test_vs1{}'.format(_n_water)]
                c12_vs1 = data.test_set['c12_test_vs1{}'.format(_n_water)]

                # VS Descriptor
                q_vs2   = data.test_set['q_test_vs2{}'.format(_n_water)]
                c6_vs2  = data.test_set['c6_test_vs2{}'.format(_n_water)]
                c12_vs2 = data.test_set['c12_test_vs2{}'.format(_n_water)]
                
                q_vs   = data.test_set['q_test_vs{}'.format(_n_water)]
                c6_vs  = data.test_set['c6_test_vs{}'.format(_n_water)]
                c12_vs = data.test_set['c12_test_vs{}'.format(_n_water)]

                if data_set.input_params['n_functions'] == 0 :
                    Hvs = np.matrix([q, c12, c6, q_vs1, q_vs2]).T   # MxNfunctionts
                elif data_set.input_params['n_functions'] == 1 :
                    Hvs = np.matrix([q, c12, c6, q_vs1, c12_vs1, q_vs2, c12_vs2]).T
                elif data_set.input_params['n_functions'] == 2 :
                    Hvs = np.matrix([q, c12, c6, q_vs1, c12_vs1, c6_vs1, q_vs2, c12_vs2, c6_vs2]).T 
                elif data_set.input_params['n_functions'] == 3 :
                    Hvs = np.matrix([q, c12, c6, q_vs1, c12_vs1, c6_vs1, q_vs2, c12_vs2, c6_vs2]).T 
                elif data_set.input_params['n_functions'] == 4 :
                    Hvs = np.matrix([q, c12, c6, q_vs1, q_vs2]).T 
                elif data_set.input_params['n_functions'] == 5 :
                    Hvs = np.matrix([q, c12, c6, q_vs1, c12_vs1, q_vs2, c12_vs2]).T 
                elif data_set.input_params['n_functions'] == 6:
                    Hvs = np.matrix([q, c12, c6, q_vs, c12_vs, c6_vs]).T 
                elif data_set.input_params['n_functions'] == 7:
                    Hvs = np.matrix([c12, c6, q_vs, c6_vs]).T 
            
                
            y_test = np.dot(Hvs, params.T).reshape(np.shape(y_ref)).T

            # OLD PARAMS
            q_old       = data_set.input_params['old_params'][0]
            sigma_old   = data_set.input_params['old_params'][1]
            epsilon_old = data_set.input_params['old_params'][2]

            C12old = (4*(epsilon_old)*(sigma_old**12))**0.5
            C6old  = (4*(epsilon_old)*(sigma_old**6))**0.5

            c_old = np.array([q_old, C12old, C6old])

            H = np.matrix([q, c12, c6]).T
            y_test_old = np.dot(H, c_old.T).T
            y_test_old = np.array(y_test_old)
            
                
            mse_old_test = np.mean(np.abs(y_test_old.T - y_test))      

            y_test_s     = np.sort(y_test,axis=0)
            y_ref_s      = np.sort(y_ref, axis=0)
            y_test_old_s = np.sort(y_test_old,axis=0)

            mse_test     = np.mean(np.power(y_test_s - y_ref_s, 2))
            mae_test_s   = np.mean(np.abs(y_ref_s - y_test_s.T))
            mae_test_old = np.mean(np.abs(y_test_old_s - y_ref_s))

            print('MSE (lrr-de) = {}'.format(mse_test))
            print('MAE (opls) = {}'.format(mae_test_old))
            print('MAE (lrr-de) = {}'.format(mae_test_s))
            print("-------")
            if plot == 'y':
                sns.set_context("paper")
                fig = plt.figure(dpi = 100)
#                 plt.rc('text', usetex=True)
#                 plt.rc('font', family='serif')
                plt.title(" TEST SET: ${0},{1}$ water molecules".format(data.flag, _n_water))
                plt.plot(y_test,label="lrr-de",marker="*")
                plt.plot(y_ref, label="ref",marker="*")
                plt.plot(y_test_old, label="opls",marker="*")
                plt.legend(loc='upper left')
                if data.flag == "energy":
                    plt.ylabel("Energy - Test set [kj]")
                elif data.flag == "force":
                    plt.ylabel("Force - Test set [kj/mol nm]")
                plt.xlabel("$N_{confs}$")   
                plt.savefig("set{}.pdf".format(count_plot), bbox_inches='tight')
                count_plot = count_plot + 1


