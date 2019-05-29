"""
    .. module:: multi_objective

    .. moduleauthor:: Matteo Peluso matteo.peluso@sns.it 
"""


class eval_lrrde():
    """
        Procedure of fitting the new data: DE --> Normal Solver/fast LOOCV
        
        Inputs:
          data_set  : data container
          params_de : params differential evolution
        Returns:
            Evaluated coefficients from the ridge regression with corresponding loocv
        
    """
    
    def __init__(self, data_set, params_de = [10, 0.7, 0.9, 50] ):
        self.data_set  = data_set
        self.params_de = params_de    
        self.log          = {}
        self.log['best']  = []
        self.log['xbest'] = []
    
        
    def eval_de(self):
        self.out = self.de(self.data_set.borders, para = self.params_de, fun = self.Cost)
        vlambda = float(self.out['xbest'][0])
        self.values, self.sigma = Normal_solver_MO(self.data_set, vlambda, "y")
        self.params = np.divide(self.values, self.sigma)
    
        print("LOOCV error {}, lambda {}, Numero Iterazioni {}, Parametri : {}". format(self.out['best'],
                                                                                     self.out['xbest'][0], 
                                                                                     self.out['N_iter'],
                                                                                      self.params.flatten()))
    
    def plt_evo(self,save=False):
        """
            Plot the evolution of the cost function
        """
        plt.figure(dpi=100)
        plt.ylabel("Cost function")
        plt.xlabel("Number of generations")
        plt.title("Cost function evolution")
        plt.plot(self.log['best'])
        if save:
            plt.savefig("EvolutionCost.pdf")
        
            

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

            Inputs:
             borders  : borders of the alpha space
             para     : params differential evolution

            Returns:
             estimation of the hyperparameter lambda
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

        eval_r2_mse(self.data_set, self.params, save=True)
