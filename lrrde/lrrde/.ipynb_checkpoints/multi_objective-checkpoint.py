"""
    .. module:: multi_objective

    .. moduleauthor:: Matteo Peluso matteo.peluso@sns.it 
"""




import numpy as np
#from tool import *
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#from normal_solver import *
#from fast_loocv import * 
#from numba import jit





# ------------------------------------ PROCEDURE DE + LRR ----------------------------------------------#

class eval_lrrde():
    """
        Procedure of fitting the new data: DE --> Normal Solver/fast LOOCV
        
        Params:
                - data_set  ---> initial data set 
                - params_de ---> parameters for the  differential evolution search
        
    """
    def fitness(self,x):
        """
            Function which specify the fitness function of our
            system necessary to be able to use the python library
            pagmo in order to evaluate a LRR-GA procedure
        """
        new_params      = Normal_solver_MO(self.data_set, x)
        loocve, mae     = fast_loocv_MO(self.data_set, new_params)
        return loocve
    
    def get_bounds(self):
        """ 
            Function which specify the borders
        """
        return ( [1e-14],[1.0])
    
    def __init__(self, data_set, params_de = [10, 0.7, 0.9, 50] ):
        self.data_set  = data_set
        self.params_de = params_de    
        self.log          = {}
        self.log['best']  = []
        self.log['xbest'] = []
    
        if self.data_set.input_params['n_functions'] in [0,1,2,7]:
            """
                It is necessary to scale the QM reference data if the
                value of the atom charge is fixed
            """
            if self.data_set.input_params['q_value'] or self.data_set.input_params['q_value'] == 0.0:
                y_c = self.data_set.input_params['q_value']*self.data_set.q
            else:
                print('insert a value for the fixed atom charge ')
            self.data_set.y_sample = self.data_set.y_sample - y_c
    
    def eval_de(self):
        self.out = self.de(self.data_set.borders, para = self.params_de, fun = self.Cost)
        vlambda = float(self.out['xbest'][0])
        self.values, self.sigma = Normal_solver_MO(self.data_set, vlambda, "y")
        self.params = np.divide(self.values, self.sigma)
    
        print("LOOCV error {}, lambda {}, Numero Iterazioni {}, Parametri :". format(np.round(self.out['best'],2),
                                                                                     np.round(self.out['xbest'][0],2), 
                                                                                     np.round(self.out['N_iter'],2)))
        self.print_res()

    def Cost(self, x):
        """
            Function which evaluate the cost function of the MO-LRR-DE
            and returns a list of lambdas

        """        
        new_params      = Normal_solver_MO(self.data_set, x)
        loocve, mae     = fast_loocv_MO(self.data_set, new_params)
        return loocve, mae 
    
    def de(self, borders, para, fun):
        """
            Differential Evolution
            
            @params borders  ---> borders of the alpha space
            @params para     ---> params differential evolution
        """
        
        # DE default params    
        # DE parameter --> population, (10 to 25)
        n = para[0]           
        # DE parameter --> scaling, (0.5 to 0.9) 
        F = para[1]           
        # DE parameter --> crossover probability
        Cr = para[2]          
        # DE parameter --> number of generations
        step_delta = para[3]
        
        # Iteration Params
        
        # Stop tolerance
        tol     = 10e-10
        k_max   = 500              # Anche meno 
        k       = 0
        delta_f = 2*tol
        
        # Initialization Number of functions 
        N_iter = 0            
        
        # Dimension of the search space
        d = np.shape(borders)[0]
       
        # definition of lower and upper bounds vectors
        Lb = borders[:,0]
        Ub = borders[:,1]
     
        # initialize the population with simple random sampling
        Lb_rep = np.tile(Lb.T, (n, 1))
        Ub_rep = np.tile(Ub.T, (n, 1))
        R_init = np.random.rand(n,d)
        
        x = Lb_rep + np.multiply((Ub_rep - Lb_rep),R_init)
        y, mae = fun(x)
        y = np.array(y)
        
        # Find the current best
        best  = np.min(y)
        I     = np.argmin(y)
        I_one = np.mean(y)
        
        # start the iterations by differential evolution
        old_best = best
        II = 0

        while delta_f > tol:
            # Prepare the matrix nd_M of the random indice
            II    = II + 1
            k     = k + 1
            index = []
            for i in range(n):
                aux  = [ii for ii in range(n)]
                aux.remove(i)
                aux  = np.random.permutation(aux)
                index.append(aux)
            nd_M = np.matrix(index) 
            
            # metric for the termination 
            if II % step_delta == True and k > 200 :
                # the algorithm performs a number of "step_delta" cycles
                I_value  = np.mean(y)
                delta_f  = abs((I_value - I_one))/(0.5*(abs(I_value) + abs(I_one)))
                I_one    = I_value 
                best     = np.min(y)
            
            # creation of the donor vector (i.e., the mutant vector)
            i_donor = nd_M[:,0:3]
            a       = np.array(x[i_donor[:,0]])
            b       = np.array(x[i_donor[:,1]])
            c       = np.array(x[i_donor[:,2]])
            v_donor = np.matrix(a + F*(b - c))
            
            if np.shape(v_donor)[0] == 1: 
                v_donor = v_donor.T
    
            # CROSSOVER (BINOMIAL SCHEME)
            R_cross =  np.random.rand(n,d)
            I_Cr    = (R_cross < Cr)
            x_new   = np.multiply(v_donor, I_Cr) + np.multiply((I_Cr == 0), x)
            
            # Selection
            y_new, mae      = fun(x_new)
            y_new           = np.array(y_new)
            v_s             = [y_new < y]
            x[tuple(v_s)]   = x_new[tuple(v_s)]
            y[tuple(v_s)]   = y_new[tuple(v_s)]
            
            # Find the current best
            best  = np.min(y)
            I     = np.argmin(y)
            xbest = x[I,:]
            my    = np.sum(y)/np.shape(y)[0]
            
            self.log['best'].append(min(y))
            self.log['xbest'].append(np.asscalar(xbest))
        N_iter = II
        print("Scaled MAE {}, loocv {}".format(min(mae), min(y)))
        return {'best':best, 'xbest':xbest,'x': x,'y': y,'N_iter': N_iter}
    
    def print_res(self):
        if self.data_set.input_params['n_vs'] == 2:
            if self.data_set.input_params['n_functions'] == 3:
                print("Atom: q  {:.4f} c12  {:.4f} c6  {:.4f}".format(self.params[0,0], self.params[0,1], self.params[0,2]))
                print("VS1: q_vs1   {:.4f} c12_vs1   {:.4f} c6_vs1 {:.4f}".format(self.params[0,3],self.params[0,4],self.params[0,5]))
                print("VS2: q_vs2   {:.4f} c12_vs2   {:.4f} c6_vs2   {:.4f}".format(self.params[0,6], self.params[0,7], self.params[0,8]))


            elif self.data_set.input_params['n_functions'] == 2:
                print("Atom: q  {:.10f} c12  {:.10f} c6  {:.10f}".format(self.data_set.q_value, self.params[0,0], self.params[0,1]))
                print("VS1: q_vs1   {:.10f} c12_vs1   {:.10f} c6_vs1 {:.10f}".format(self.params[0,2],self.params[0,3],self.params[0,4]))
                print("VS2: q_vs2   {:.10f} c12_vs2   {:.10f} c6_vs2   {:.10f}".format(self.params[0,5],self.params[0,6], self.params[0,7]))

            elif self.data_set.input_params['n_functions'] == 1:
                print("Atom: q  {:.10f} c12  {:.10f} c6  {:.10f}".format(self.data_set.q_value, self.params[0,0], self.params[0,1]))
                print("VS1: q_vs1   {:.10f} c12_vs1   {:.10f} c6_vs1 {:.10f}".format(self.params[0,2],self.params[0,3],0.0))
                print("VS2: q_vs2   {:.10f} c12_vs2   {:.10f} c6_vs2   {:.10f}".format(self.params[0,4],self.params[0,5], 0.0)  )          

            elif self.data_set.input_params['n_functions'] == 0:
                print("Atom: q  {:.10f} c12  {:.10f} c6  {:.10f}".format(self.data_set.q_value, self.params[0,0], self.params[0,1]))
                print("VS1: q_vs1   {:.10f} c12_vs1   {:.10f} c6_vs1 {:.10f}".format(self.params[0,2], 0.0, 0.0))
                print("VS2: q_vs2   {:.10f} c12_vs2   {:.10f} c6_vs2   {:.10f}".format(self.params[0,3], 0.0, 0.0)   )

            elif self.data_set.input_params['n_functions'] == 4:
                print("Atom: q  {:.10f} c12  {:.10f} c6  {:.10f}".format(self.params[0,0], self.params[0,1], self.params[0,2]))
                print("VS1: q_vs1   {:.10f} c12_vs1   {:.10f} c6_vs1 {:.10f}".format(self.params[0,3], 0.0, 0.0))
                print("VS2: q_vs2   {:.10f} c12_vs2   {:.10f} c6_vs2   {:.10f}".format(self.params[0,4], 0.0, 0.0))
                
            elif self.data_set.input_params['n_functions'] == 5:
                print("Atom: q  {:.10f} c12  {:.10f} c6  {:.10f}".format(self.params[0,0], self.params[0,1], self.params[0,2]))
                print("VS1: q_vs1   {:.10f} c12_vs1   {:.10f} c6_vs1 {:.10f}".format(self.params[0,3], self.params[0,4], 0.0))
                print("VS2: q_vs2   {:.10f} c12_vs2   {:.10f} c6_vs2   {:.10f}".format(self.params[0,5], self.params[0,6], 0.0))
            elif self.data_set.input_params['n_functions'] == 6:
                print(self.params)
            elif self.data_set.input_params['n_functions'] == 7:
                print(self.params)


        else:
            # Case of no VS
            if self.data_set.input_params['n_functions'] == 3:
                print("Atom: q  {:.10f} c12  {:.10f} c6  {:.10f}".format(self.params[0,0], self.params[0,1], self.params[0,2]))
            elif self.data_set.input_params['n_functions'] == 2:
                print("Atom: q  {:.10f} c12  {:.10f} c6  {:.10f}".format(self.data_set.q_value, self.params[0,0], self.params[0,1]))

        eval_r2_mse(self.data_set, self.params)
