import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

def gen_vs(topology, coords, index_at, n_vs, len_vs = 0.075):
    """
        Function which takes as input a set of coordinates and for each 
        generate a new set with  @n_vs Virtual Site around the @index_at
        atom at the distance of @len_vs
        
        @len_vs: length value of the VS 
        @n_vs:   number of virtual site to be added   
        
        The @n_vs coordinates are the last @n_vs of each configuration
    """
    
    new_coord, coord_vs, new_topol = [], [], []
        
    if n_vs == 2:
        """
            Linear geometry
        """
        new_topol.append(np.hstack([topology[0], np.zeros(n_vs)]))   # Initial Charge VS
        new_topol.append(np.hstack([topology[1], np.zeros(n_vs)  ]))   # Initial C6 VS1
        new_topol.append(np.hstack([topology[2], np.zeros(n_vs) ]))   # Initial C12 VS1


        for coord in coords:
            atom    = coord[index_at, :]
            # First Virtual Site
            vs_1    = [ atom[0] - len_vs, atom[1], atom[2] ]
            vs_1    = np.array(vs_1)
            # Second Virtual Site
            vs_2    = [ atom[0] + len_vs, atom[1], atom[2]]
            vs_2    = np.array(vs_2)
            new_coord.append(np.vstack([coord, vs_1, vs_2]))       
        
        return   np.array(new_topol), np.array(new_coord)    


def standardize_data(data):
    """
        Function which takes as input a data array and
        returns his mean, standard deviation and
        the data divided by his std
    """
    mean     = np.mean(data)
    sigma    = np.std(data)
    data_std = np.divide(data, sigma) 
    return mean, sigma, data_std

def leverage(data_set, flag = "novs"):
    """
        Function which leverage data
    """
    # QM Data
    y    = data_set.y_sample
    # Weight matrix
    v    = data_set.w_vector  
    W    = data_set.w_matrix 
    
    sigma, H = sigma_h(data_set)

    n    = np.size(y,0)
    V    = np.tile(v,(np.size(H,1),1))
    
    H = np.dot(H,V)
    
    mean = np.mean(H)
    Mean = np.ones(np.shape(H))
    dev  = H - Mean
    num  = np.sum(np.power(dev,2),1)
    den  = np.sum(num)
    
    l = 1/n + num/den
    return l


def eval_r2_mse(data_set, params):
    """
        Evaluate the R2/MSE/MAE of the output
        and compares it with the old values
        
         @params data_set
         @params params    ---> new params
    """
    y_ref    = data_set.y_sample

    # Non scaled Descriptor
    q   = data_set.q
    c6  = data_set.c6
    c12 = data_set.c12
    
    # Old Params
    q_old       = data_set.input_params['old_params'][0]
    sigma_old   = data_set.input_params['old_params'][1]
    epsilon_old = data_set.input_params['old_params'][2]

    C12old = (4*(epsilon_old)*(sigma_old**12))**0.5
    C6old  = (4*(epsilon_old)*(sigma_old**6))**0.5
    
    c_old = np.array([q_old, C12old, C6old])
    H = np.matrix([q, c12, c6]).T
    y_test_old = np.dot(H, c_old.T).reshape(np.shape(y_ref)).T
    y_test_old = np.array(y_test_old)
    
    if data_set.input_params['n_functions'] in [0,1,2]:
        y_c        = data_set.input_params['q_value']*data_set.q
        y_c        = y_c.reshape(np.shape(y_test_old))
        y_test_old = y_test_old - y_c
    
        
    if data_set.input_params['n_vs'] == 0:
        if data_set.input_params['n_functions'] == 3 :
            Ht = np.matrix([q, c12, c6]).T   # MxN functionts
        elif data_set.input_params['n_functions'] == 0 :
            Ht = np.matrix([c12, c6]).T
        
    elif data_set.input_params['n_vs'] == 2:

        # VS Descriptor
        q_vs1   = data_set.q_vs1
        c6_vs1  = data_set.c6_vs1
        c12_vs1 = data_set.c12_vs1
        
        q_vs   = data_set.q_vs
        c6_vs  = data_set.c6_vs
        c12_vs = data_set.c12_vs

        # VS Descriptor
        q_vs2   = data_set.q_vs2
        c6_vs2  = data_set.c6_vs2
        c12_vs2 = data_set.c12_vs2
                
        if data_set.nfunctions == 0:      
            Ht = np.matrix([c12, c6, q_vs1, q_vs2]).T   # MxNfunctionts                
        elif data_set.nfunctions == 1:            
            Ht = np.matrix([c12, c6, q_vs1,c12_vs1, q_vs2,c12_vs2]).T   # MxNfuncti\onts            
        elif data_set.nfunctions == 2:
            Ht = np.matrix([c12, c6, q_vs1, c12_vs1, c6_vs1, q_vs2, c12_vs2, c6_vs2]).T   # MxNfunctionts            
        elif data_set.nfunctions == 3:
            Ht = np.matrix([q, c12, c6, q_vs1, c12_vs1, c6_vs1, q_vs2, c12_vs2, c6_vs2]).T   # MxNfunctionts            
        elif data_set.nfunctions == 4:
            Ht = np.matrix([q, c12, c6, q_vs1, q_vs2]).T   # MxNfunctionts
        elif data_set.nfunctions == 5:
            Ht = np.matrix([q, c12, c6, q_vs1, c12_vs1, q_vs2, c12_vs2]).T   # MxNfunctionts
        elif data_set.nfunctions == 6:
            Ht = np.matrix([q, c12, c6, q_vs, c12_vs, c6_vs]).T   # MxNfunctionts
        elif data_set.nfunctions == 7:
            Ht = np.matrix([c12, c6, q_vs, c6_vs]).T   # MxNfunctionts

                    
    # Eval metrics lrr-de
    y_test    = np.dot(Ht, params.T).reshape(np.shape(y_ref)).T
    
    r2_lrrde  =  r2_score(y_ref, y_test)
    mse_lrrde = np.mean(np.power(y_test - y_ref, 2))
    mae_lrrde = np.mean(np.abs(y_test - y_ref.T))
    print("R2 score:  lrr-de", r2_lrrde)
    print("MSE score: lrr-de", mse_lrrde)
    print("MAE score: lrr-de", mae_lrrde)

    # Eval metrics old params
    r2_old =  r2_score(y_ref, y_test_old)
    mse_old = np.mean(np.power(y_test_old - y_ref, 2))
    mae_old = np.mean(np.abs(y_test_old - y_ref))
    print("R2 score  old params:", r2_old )
    print("MSE score old params:", mse_old)
    print("MAE score old params:", mae_old)
    
    fig = plt.figure(dpi = 100)
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
    sns.set_context("paper")
    plt.title("Sorted input -- output values")
    plt.plot(np.sort(y_ref,axis=0), "r-", label="ref" )
    plt.plot(np.sort(y_test,axis=0),"b-", label="lrr-de" )
    plt.plot(np.sort(y_test_old,axis=0),"g-",  label="opls")
    #plt.tight_layout()
    plt.legend(loc='upper left')
    
    sns.set_context("paper")
    fig = plt.figure(dpi = 100)
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
    plt.title("Energy contribution")
    plt.plot(y_ref[0:data_set.input_params['n_train']*2],  "r-", label="ref" )
    plt.plot(y_test[0:data_set.input_params['n_train']*2], "b-", label="lrr-de" )
    plt.plot(y_test_old[0:data_set.input_params['n_train']*2], "g-",  label="opls")
    plt.legend(loc='upper left')      
    plt.xlabel("$N_{conf}$")
    plt.ylabel("Energy [Kj]")
    

    f = plt.figure(dpi=100)
    sns.set_context("paper")
    plt.title("Force contribution")
    plt.plot(y_test_old[data_set.input_params['n_train']*2:], "g-",  label="opls")
    plt.plot(y_ref[data_set.input_params['n_train']*2:],  "r-", label="ref" )
    plt.plot(y_test[data_set.input_params['n_train']*2:], "b-", label="lrr-de" )
    
    plt.legend(loc='upper left')
    plt.xlabel("$N_{conf}*3$")
    plt.ylabel("Force [Kj/mol nm]")
    plt.show() 
        
    return r2_lrrde, mse_lrrde
def sigma_h(data_set):
    """
        given a data set in input returns the H matrix and his deviation sigma
    """
    # Non scaled Descriptor
    q   = data_set.q
    c6  = data_set.c6
    c12 = data_set.c12

    # Scaled Descriptor
    mean_q,   sigma_q,   data_std_q   = standardize_data(q)
    mean_c6,  sigma_c6,  data_std_c6  = standardize_data(c6)
    mean_c12, sigma_c12, data_std_c12 = standardize_data(c12)
    
    if data_set.input_params['n_vs'] == 0:
        if data_set.nfunctions == 3:
            sigma  = np.array([sigma_q, sigma_c12, sigma_c6])    
            H      = np.matrix([data_std_q, data_std_c12, data_std_c6]).T
        elif data_set.nfunctions == 0:
            sigma  = np.array([sigma_c12, sigma_c6])    
            H      = np.matrix([data_std_c12, data_std_c6]).T            
    
    elif data_set.input_params['n_vs'] == 2:
    # VS Descriptor
        q_vs1   = data_set.q_vs1
        c6_vs1  = data_set.c6_vs1
        c12_vs1 = data_set.c12_vs1

        # VS Descriptor
        q_vs2   = data_set.q_vs2
        c6_vs2  = data_set.c6_vs2
        c12_vs2 = data_set.c12_vs2

        mean_q_vs1,   sigma_q_vs1,   data_std_q_vs1   = standardize_data(q_vs1)
        mean_c6_vs1,  sigma_c6_vs1,  data_std_c6_vs1  = standardize_data(c6_vs1)
        mean_c12_vs1, sigma_c12_vs1, data_std_c12_vs1 = standardize_data(c12_vs1)

        mean_q_vs2,   sigma_q_vs2,   data_std_q_vs2   = standardize_data(q_vs2)
        mean_c6_vs2,  sigma_c6_vs2,  data_std_c6_vs2  = standardize_data(c6_vs2)
        mean_c12_vs2, sigma_c12_vs2, data_std_c12_vs2 = standardize_data(c12_vs2)
        
        q_vs   = data_set.q_vs
        c6_vs  = data_set.c6_vs
        c12_vs = data_set.c12_vs
        
        mean_q_vs,   sigma_q_vs,   data_std_q_vs   = standardize_data(q_vs)
        mean_c6_vs,  sigma_c6_vs,  data_std_c6_vs  = standardize_data(c6_vs)
        mean_c12_vs, sigma_c12_vs, data_std_c12_vs = standardize_data(c12_vs)
        
        
        
        
        if data_set.nfunctions == 7:
            sigma  = np.array([sigma_c12, sigma_c6, \
                               sigma_q_vs, sigma_c6_vs])
            
            H      = np.matrix([ data_std_c12, data_std_c6, \
                           data_std_q_vs, data_std_c6_vs]).T
            
        elif data_set.nfunctions == 6:
            sigma  = np.array([sigma_q, sigma_c12, sigma_c6, \
                               sigma_q_vs, sigma_c12_vs, sigma_c6_vs])
            
            H      = np.matrix([data_std_q, data_std_c12, data_std_c6, \
                           data_std_q_vs, data_std_c12_vs, data_std_c6_vs]).T
            
        elif data_set.nfunctions == 5:            
            sigma  = np.array([sigma_q, sigma_c12, sigma_c6, \
                               sigma_q_vs1, sigma_c12_vs1,\
                               sigma_q_vs2, sigma_c12_vs2])
            
            H      = np.matrix([data_std_q, data_std_c12, data_std_c6, \
                           data_std_q_vs1, data_std_c12_vs1, \
                           data_std_q_vs2, data_std_c12_vs2]).T
    
        elif data_set.nfunctions == 4:            
            sigma  = np.array([sigma_q, sigma_c12, sigma_c6, \
                               sigma_q_vs1, \
                               sigma_q_vs2])
            
            H      = np.matrix([data_std_q, data_std_c12, data_std_c6, \
                           data_std_q_vs1, \
                           data_std_q_vs2]).T 
            
        elif data_set.nfunctions == 3:
            sigma  = np.array([sigma_q, sigma_c12, sigma_c6, \
                               sigma_q_vs1, sigma_c12_vs1, sigma_c6_vs1, \
                               sigma_q_vs2, sigma_c12_vs2, sigma_c6_vs2])
            
            H      = np.matrix([data_std_q, data_std_c12, data_std_c6, \
                                data_std_q_vs1, data_std_c12_vs1, data_std_c6_vs1,\
                                 data_std_q_vs2, data_std_c12_vs2, data_std_c6_vs2]).T   # MxNfunctionts    
             
        elif data_set.nfunctions == 2:
            sigma  = np.array([sigma_c12, sigma_c6, \
                               sigma_q_vs1, sigma_c12_vs1, sigma_c6_vs1,\
                               sigma_q_vs2, sigma_c12_vs2, sigma_c6_vs2])
            
            H      = np.matrix([data_std_c12, data_std_c6, \
                           data_std_q_vs1, data_std_c12_vs1, data_std_c6_vs1,\
                           data_std_q_vs2, data_std_c12_vs2, data_std_c6_vs2]).T   # MxNfunctionts
            
        elif data_set.nfunctions == 1:                       
            sigma  = np.array([sigma_c12, sigma_c6, \
                               sigma_q_vs1, sigma_c12_vs1, \
                               sigma_q_vs2, sigma_c12_vs2])
            
            H      = np.matrix([data_std_c12, data_std_c6, \
                           data_std_q_vs1, data_std_c12_vs1, \
                           data_std_q_vs2, data_std_c12_vs2]).T
            
        elif data_set.nfunctions == 0:            
            sigma  = np.array([sigma_c12, sigma_c6, \
                               sigma_q_vs1, \
                               sigma_q_vs2])
            
            H      = np.matrix([data_std_c12, data_std_c6, \
                                data_std_q_vs1, \
                                data_std_q_vs2]).T 
    return sigma, H
