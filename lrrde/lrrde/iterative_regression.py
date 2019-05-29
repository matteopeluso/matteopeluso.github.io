from out_function import *
from parse_input  import *
from read         import *
import numpy as np
import seaborn as sns

def cost_function(X, y, theta):
    """
    cost_function(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    """
    ## number of training examples
    m = len(y) 
    
    ## Calculate the cost with the given parameters
    J = np.sum((X.T.dot(theta)-y)**2)/2/m
    return J
class iterative_regression():
    """
        pancakes
    """
    
    def __init__(self, data_set, iterations, alpha, flag, flag_vs):
        self.data_set   = data_set
        self.iterations = iterations
        self.alpha      = alpha
        self.flag       = flag
        self.flag_vs    = flag_vs
        
        
        self.cost_history = [0] * self.iterations
        self.q_h          = [0] * self.iterations
        self.c6_h         = [0] * self.iterations
        self.c12_h        = [0] * self.iterations
        
        if self.flag_vs   == True:
            self.q_h_vs   = [0] * self.iterations
            self.c6_h_vs  = [0] * self.iterations
            self.c12_h_vs = [0] * self.iterations

    def do_lin_regression(self):
        if self.flag not in ['old', 'rand']:
            print("insert a proper flag for the iterative linear regression {}".format(self.flag))
            return -1

        if self.flag_vs   == True:
            self.X   = np.array([
                    (self.data_set.q      - np.mean(self.data_set.q)      )/np.std(self.data_set.q),
                    (self.data_set.c12    - np.mean(self.data_set.c12)    )/np.std(self.data_set.c12),
                    (self.data_set.c6     - np.mean(self.data_set.c6)     )/np.std(self.data_set.c6),
                    (self.data_set.q_vs   - np.mean(self.data_set.q_vs)   )/np.std(self.data_set.q_vs),
                    (self.data_set.c12_vs - np.mean(self.data_set.c12_vs) )/np.std(self.data_set.c12_vs),
                    (self.data_set.c6_vs  - np.mean(self.data_set.c6_vs)  )/np.std(self.data_set.c6_vs)])
            
            self.X_ns   = np.array([
                    (self.data_set.q     ),
                    (self.data_set.c12   ),
                    (self.data_set.c6    ),
                    (self.data_set.q_vs  ),
                    (self.data_set.c12_vs),
                    (self.data_set.c6_vs )])
            
            self.std_ = np.array([np.std(self.data_set.q), np.std(self.data_set.c12), np.std(self.data_set.c6),
                                 np.std(self.data_set.q_vs), np.std(self.data_set.c12_vs), np.std(self.data_set.c6_vs)])
        else:
            self.X   = np.array([
                    (self.data_set.q    - np.mean(self.data_set.q)  )/np.std(self.data_set.q),
                    (self.data_set.c12  - np.mean(self.data_set.c12))/np.std(self.data_set.c12),
                    (self.data_set.c6   - np.mean(self.data_set.c6) )/np.std(self.data_set.c6)])

            self.X_ns = np.array([self.data_set.q, self.data_set.c12, self.data_set.c6])
            
            self.std_ = np.array([np.std(self.data_set.q), np.std(self.data_set.c12), np.std(self.data_set.c6)])

        self.y    = np.array(self.data_set.y_sample)
        
        q_old       = self.data_set.input_params['old_params'][0]
        sigma_old   = self.data_set.input_params['old_params'][1]
        epsilon_old = self.data_set.input_params['old_params'][2]

        C12old = (4*(epsilon_old)*(sigma_old**12))**0.5
        C6old  = (4*(epsilon_old)*(sigma_old**6))**0.5

        self.spezia = np.array([q_old, C12old, C6old])
        
        if self.flag == 'old':
            if self.flag_vs   == True:
                self.spezia = np.append(self.spezia, np.zeros(3))
                self.theta  = self.spezia*self.std_
            else:
                self.theta = self.spezia*self.std_
        
        elif self.flag == 'rand':
            if self.flag_vs   == True:
                self.spezia = np.append(self.spezia, np.zeros(3))
                self.theta = np.random.rand(6)
            else:
                self.theta = np.random.rand(3)
        
        self.gradient_descent()
        
        
    def cost_function(self):
        """
        cost_function(X, y, theta) computes the cost of using theta as the
        parameter for linear regression to fit the data points in X and y
        """
        ## number of training examples
        m = len(self.y) 

        ## Calculate the cost with the given parameters
        J = np.sum((self.X.T.dot(self.theta)- self.y)**2)/2/m
        return J
    
    def gradient_descent(self):
        """
        gradient_descent Performs gradient descent to learn theta
        theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
        taking num_iters gradient steps with learning rate alpha
        """
        m = len(self.y)

        for iteration in range(self.iterations):
            hypothesis = self.X.T.dot(self.theta)
            loss       = hypothesis - self.y
            gradient   = self.X.dot(loss)/m

            self.q_h[iteration]          = self.theta[0]
            self.c6_h[iteration]         = self.theta[2]
            self.c12_h[iteration]        = self.theta[1]
            
            if self.flag_vs == True:
                self.q_h_vs[iteration]          = self.theta[3]
                self.c6_h_vs[iteration]         = self.theta[5]
                self.c12_h_vs[iteration]        = self.theta[4]

            self.theta = self.theta - self.alpha*gradient
            cost       = self.cost_function()
            self.cost_history[iteration] = cost

        self.params = self.theta/self.std_

    def plot_results(self):
        """
            Function which:
            1) -- prints the MAE between the new params
            and the QM values
            2) -- plots the parameter's space
        """
        print("MAE spezia     --- QM {}".format(np.mean(np.abs(np.dot(self.X_ns.T, self.spezia) - self.y))))
        print("MAE regression --- QM {}".format(np.mean(np.abs(np.dot(self.X_ns.T, self.params) - self.y))))
        
        sns.set_context("paper")
        fig = plt.figure(dpi = 120)
        plt.subplots_adjust(wspace=0.4,hspace = 0.4)
        plt.subplot(131)
        plt.title("Q space")
        plt.plot(self.q_h/np.std(self.data_set.q))
        plt.subplot(132)
        plt.title("C6 space")
        plt.plot(self.c6_h/np.std(self.data_set.c6))
        plt.subplot(133)
        plt.title("C12 space")
        plt.plot(self.c12_h/np.std(self.data_set.c12))
        
        if self.flag_vs == True:
            fig = plt.figure(dpi = 120)
            plt.subplots_adjust(wspace=0.4,hspace = 0.4)
            plt.subplot(131)
            plt.title("Q space")
            plt.plot(self.q_h_vs/np.std(self.data_set.q_vs))
            plt.subplot(132)
            plt.title("C6 space")
            plt.plot(self.c6_h_vs/np.std(self.data_set.c6_vs))
            plt.subplot(133)
            plt.title("C12 space")
            plt.plot(self.c12_h_vs/np.std(self.data_set.c12_vs))
        plt.show()
