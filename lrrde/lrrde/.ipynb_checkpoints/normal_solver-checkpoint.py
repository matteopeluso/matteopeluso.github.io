#from tool import *
from numba import jit
    
def Normal_solver_MO(data_set, vlambda, flag = "n"):
    """
        This function takes as input a data_set and a set of lambdas 
        and evaluates the Multi Objective Normal Equation as
        the equation reported on the article
    """
    
    sigma, H = sigma_h(data_set)
            
    # QM Reference data
    y    = data_set.y_sample

    # Weight matrix
    v = data_set.w_vector
    W = data_set.w_matrix
        
    values  = []
    
    M      = np.size(H,0)
    N_func = np.size(H,1)
    I      = np.eye(N_func, N_func)
    #n      = np.size(vlambda,0)
   
    
    if isinstance(vlambda, list):
        for l in vlambda:
            values.append(solve_normal_eq(H,W,M,l,I,y))  # Normal Equation Multi 
            
    elif isinstance(vlambda, np.ndarray):
        for l in vlambda:
            l = np.asscalar(l)
            values.append(solve_normal_eq(H,W,M,l,I,y))  # Normal Equation Multi Objective
    else:
        l = vlambda
        values.append(solve_normal_eq(H,W,M,l,I,y))  # Normal Equation Multi 
        
    values = np.array(values)
    dim    = np.shape(values) 
    values = np.resize(values,(dim[0],dim[-1]))
    if flag == "n":
        return values
    else:
        return values, sigma

@jit(nopython=True,parallel = True,fastmath = True)
def solve_normal_eq(H,W,M,l,I,y):
    return np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(H.T,W),H) + 2*M*l*I), H.T),W),y)
