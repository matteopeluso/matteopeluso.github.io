from numba import jit
    
def Normal_solver_MO(data_set, vlambda, flag = "n"):
    """
        This function takes as input a data_set and a set of lambdas 
        and evaluates the Multi Objective Normal Equation as
        the equation reported on the article

        Inputs:
            data_set: data container
            vlambda: hyperparameter for regularization
        Returns:
            evaluation of the ridge regression
    """
            
    # QM Reference data
    y    = data_set.y_train
    H    = data_set.x_train
        
    values  = []
    
    N_func = np.size(H,1)
    I      = np.eye(N_func, N_func)
    
    if isinstance(vlambda, list):
        for l in vlambda:
            values.append(solve_normal_eq(H,l,I,y))  # Normal Equation Multi 
            
    elif isinstance(vlambda, np.ndarray):
        for l in vlambda:
            l = np.asscalar(l)
            values.append(solve_normal_eq(H,l,I,y))  # Normal Equation Multi Objective
    else:
        l = vlambda
        values.append(solve_normal_eq(H,l,I,y))  # Normal Equation Multi 
        
    values = np.array(values)
    dim    = np.shape(values) 
    values = np.resize(values,(dim[0],dim[-1]))
    if flag == "n":
        return values
    else:
        return values, data_set.ip['sigma']

@jit(nopython=True,parallel = True,fastmath = True)
def solve_normal_eq(H,l,I,y):
    """
        Evaluation of the Normal Equation

        Inputs
            H:  training set
            l:  hyperparameter for the regularization
            I:  Identity matrix
            y:  reference

        Returns
            Estimate C of the ridge regression
    """
    return np.dot(np.dot(np.linalg.inv(np.dot(H.T,H) + l*I), H.T),y)
