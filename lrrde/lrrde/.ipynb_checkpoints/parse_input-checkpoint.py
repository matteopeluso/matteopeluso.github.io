/"""
    .. module:: parse_input

    :platform: Unix
    :synopsis: Parses forces and energy input files and build a training set

    .. moduleauthor:: Matteo Peluso
"""

import numpy as np
#from lrr_de_constant import *
import matplotlib.pyplot as plt
import os
#from read import *
#from distances import *
#from tool import *

class parser():
    """
    Class parser

       The parser takes as input files which names are already
       decided
            forces: forces128.txt where 128 will be the number of water molecules of the spherical cluster taken in account

            energies: complexation128.txt the term complexation refers to the fact that the QM calculation have been evaluated with a Counterpoise correction

            coordinates: coordinates128.txt coordinates of the cluster centered on the ion

            topology: topology128.txt topology of the system Charge-sigma-epsilon

       Input parameters:
            input_params --> collection of input parameters

            flag_files   --> flag which activate the parsing for a certain type of file
      
       Returns:
            collection of parsed data
            
    """
    def __init__(self, input_params, flag_files):
        """
            Initialization
        """
        self.ip = input_params
        self.flag_files   = flag_files
    
    def parse(self):
        try:
            os.chdir(self.ip['file_dir'])
        except:
            print("change path for the files to be parsed")
        for _n_water in self.ip['N_water']:
            
            _n_atoms = _n_water*3 + 1
            
            if self.flag_files['topology']:
                self.ip['topol{}'.format(_n_water)]   = read_topol("topology{}.txt".format(_n_water))
                
            if self.flag_files['energy']:
                self.ip['energy{}'.format(_n_water)]     = read_ener("complexation{}.txt".format(_n_water), self.ip['counterpoise'])
                
            if self.flag_files['force']:
                self.ip['force{}'.format(_n_water)]      = read_forces("forces{}.txt".format(_n_water), _n_atoms)
                
            if self.flag_files['coordinate']:
                self.ip['coordinates{}'.format(_n_water)] = read_coord("coordinates{}.txt".format(_n_water), _n_atoms)
                
        return self.ip

    def eval_distances(self):
        """
        Method with which is possible to evaluate the distance descriptor

        Input Parameters:
            self.coordinates

        Returns:
            a vector of Eucleudian distances and a matrix of distances
        """
        self.ip = self.parse()
        for _n_water in self.ip['N_water']:
            _n_atoms = _n_water*3 + 1
            
            if self.ip['n_vs'] == 0:
                
                self.ip['d_vet{}'.format(_n_water)] = distance_ion(mat = self.ip['coordinates{}'.format(_n_water)], 
                                                                             index  = 1,
                                                                             n_conf = self.ip['n_conf'],
                                                                             n_atom = _n_atoms,
                                                                             n_vs   = 0)
                
                self.ip['d_mat{}'.format(_n_water)] = distance_ion_mat(mat = self.ip['coordinates{}'.format(_n_water)], 
                                                                             index  = 1,
                                                                             n_conf = self.ip['n_conf'],
                                                                             n_atom = _n_atoms,
                                                                             n_vs   = 0)
                
            elif self.ip['n_vs'] == 2:
                # Generates the VS
                self.ip['topol{}'.format(_n_water)], self.ip['coordinates{}'.format(_n_water)] = gen_vs(self.ip['topol{}'.format(_n_water)], 
                                                                        self.ip['coordinates{}'.format(_n_water)],
                                                                        index_at = 0,
                                                                        n_vs     = self.ip['n_vs'],
                                                                        len_vs   = self.ip['len_vs'])
                
                self.ip['d_vet{}'.format(_n_water)], self.ip['d_vet_vs1{}'.format(_n_water)], self.ip['d_vet_vs2{}'.format(_n_water)] = distance_ion( 
                    mat = self.ip['coordinates{}'.format(_n_water)],     
                    n_conf = self.ip['n_conf'],
                    index  = 2,
                    n_atom = _n_atoms,
                    n_vs   = self.ip['n_vs'])
                
                self.ip['d_mat{}'.format(_n_water)], self.ip['d_mat_vs1{}'.format(_n_water)], self.ip['d_mat_vs2{}'.format(_n_water)] = distance_ion_mat(
                    mat = self.ip['coordinates{}'.format(_n_water)],
                    n_conf = self.ip['n_conf'],
                    index   = 2,
                    n_atom  = _n_atoms,
                    n_vs    = self.ip['n_vs'])
        
        return self.ip
        
    

    
class input_setup():
    """
        This function take as input the parsed input files, or forces or energies, the evaluated
        distances, the size of training and test set and returns a class 
        
        Input Parameters:

           * input_params['topol ']       = output of parse_input for topology 
           * input_params['forces']       = output of parse_input for forces
           * input_params['energy']       = output of parse_input for energy
           * input_params['dis_vet']      = output of distance_ion
           * input_params['dis_mat']      = output of distance_ion_mat
           * input_params['N_train']      = Dimension of the training set
           * input_params['N_test']       = Dimension of the test set
           * input_params['data_type']    = Type of the class output, or 'energy' or 'force'
           * input_params['weight']       = Weight of the class output, default 1
           * input_params['borders']      = Border of the search space, default [10e-15, 10e-1]
           * input_params['n_functions']  = Number of function 
           * input_params['name']         = Name of the virtual site
           * input_params['flag']         = Flag for test set and training set

        Returns:
            Evaluate from the coordinates and the topology the energies and the forces of each system that are going to be our descriptor through the procedure
    """
    
    def __init__(self, input_params, flag, weight):
        
        self.ip           = input_params
        self.N_train      = input_params['n_train']
        self.N_test       = input_params['n_test']
        self.flag         = flag
        self.t_s          = {}  # Training Set
        self.test_set     = {}
        try:
            self.ip['weight32']       = weight[0]
            self.ip['weight128']      = weight[1]
        except:
            self.ip['weight32']       = 1
            self.ip['weight128']      = 1

    
        if self.flag not in ['energy', 'force']:
            print("Choose a proper data type for the input setup between energy or force: {}".format(self.flag))
            return -1

        if input_params['n_train'] < input_params['n_test'] and input_params['n_train'] > 0:
            print("n_train not consistent:    {}".format(input_params['n_train']))
            return -1
    
        if self.flag == 'energy':
            self.t_s = self.initialize_energy()
            self.t_s = self.eval_energy(flag = "train")
            self.test_set     = self.eval_energy(flag = "test") 
            
        elif self.flag == 'force':
            self.t_s = self.initialize_force()
            self.t_s = self.eval_force(flag = "train")
            self.test_set     = self.eval_force(flag = "test") 
    
    def initialize_energy(self):
        """
            Initialize energy set
        """
        if self.ipip['n_vs'] == 0:            
            if self.ipip['counterpoise'] == False:
                for _n_water in self.ip['N_water']:
                    self.t_s['y_sample{}'.format(_n_water)]     = self.ip['energy{}'.format(_n_water)][0 : self.N_train   , 0] 
                    self.t_s['weights{}'.format(_n_water)]      = self.ip['weight{}'.format(_n_water)]/np.std(self.ip['energy{}'.format(_n_water)][: , 0])
                    self.t_s['x_sample{}'.format(_n_water)]     = self.ip['d_vet{}'.format(_n_water)][0 : self.N_train , :]
                    self.t_s['x_test{}'.format(_n_water)]       = self.ip['d_vet{}'.format(_n_water)][self.N_train : self.N_train + self.N_test , :]
            else:                    
                for _n_water in self.ip['N_water']:
                    self.t_s['y_sample{}'.format(_n_water)]     = self.ip['energy{}'.format(_n_water)][0 : self.N_train   , 1] 
                    self.t_s['weights{}'.format(_n_water)]      = self.ip['weight{}'.format(_n_water)]/np.std(self.ip['energy{}'.format(_n_water)][: , 1])
                    self.t_s['energy{}'.format(_n_water)]       = self.ip['energy{}'.format(_n_water)]    
                    self.t_s['y_test{}'.format(_n_water)]       = self.ip['energy{}'.format(_n_water)][self.N_train : self.N_train + self.N_test , 0]
                    self.t_s['x_sample{}'.format(_n_water)]     = self.ip['d_vet{}'.format(_n_water)][0 : self.N_train , :]
                    self.t_s['x_test{}'.format(_n_water)]       = self.ip['d_vet{}'.format(_n_water)][self.N_train : self.N_train + self.N_test , :]
                    
        elif self.ip['n_vs'] == 2:
            if self.ip['counterpoise'] == False:
                for _n_water in self.ip['N_water']:
                    self.t_s['y_sample{}'.format(_n_water)]     = self.ip['energy{}'.format(_n_water)][0 : self.N_train] 
                    self.t_s['y_sample_vs1{}'.format(_n_water)] = self.ip['energy{}'.format(_n_water)][0 : self.N_train] 
                    self.t_s['y_sample_vs2{}'.format(_n_water)] = self.ip['energy{}'.format(_n_water)][0 : self.N_train] 
                    
                    self.t_s['weights{}'.format(_n_water)]      = self.ip['weight{}'.format(_n_water)]/np.std(self.ip['energy{}'.format(_n_water)][:])
                    self.t_s['weights_vs1{}'.format(_n_water)]  = self.ip['weight{}'.format(_n_water)]/np.std(self.ip['energy{}'.format(_n_water)][:])
                    self.t_s['weights_vs2{}'.format(_n_water)]  = self.ip['weight{}'.format(_n_water)]/np.std(self.ip['energy{}'.format(_n_water)][:])
                    
                    self.t_s['x_sample{}'.format(_n_water)]     = self.ip['d_vet{}'.format(_n_water)][0 : self.N_train , :]
                    self.t_s['x_sample_vs1{}'.format(_n_water)] = self.ip['d_vet_vs1{}'.format(_n_water)][0 : self.N_train , :]
                    self.t_s['x_sample_vs2{}'.format(_n_water)] = self.ip['d_vet_vs2{}'.format(_n_water)][0 : self.N_train , :]
                    
                    self.t_s['y_test{}'.format(_n_water)]       = self.ip['energy{}'.format(_n_water)][self.N_train : self.N_train + self.N_test]
                    self.t_s['y_test_vs1{}'.format(_n_water)]   = self.ip['energy{}'.format(_n_water)][self.N_train : self.N_train + self.N_test]
                    self.t_s['y_test_vs2{}'.format(_n_water)]   = self.ip['energy{}'.format(_n_water)][self.N_train : self.N_train + self.N_test]
                    
                    self.t_s['x_test{}'.format(_n_water)]       = self.ip['d_vet{}'.format(_n_water)][self.N_train : self.N_train + self.N_test , :]
                    self.t_s['x_test_vs1{}'.format(_n_water)]   = self.ip['d_vet_vs1{}'.format(_n_water)][self.N_train : self.N_train + self.N_test , :]
                    self.t_s['x_test_vs2{}'.format(_n_water)]   = self.ip['d_vet_vs2{}'.format(_n_water)][self.N_train : self.N_train + self.N_test , :]
                    
            else:                    
                for _n_water in self.ip['N_water']:
                    self.t_s['y_sample{}'.format(_n_water)]     = self.ip['energy{}'.format(_n_water)][0 : self.N_train   , 1]
                    self.t_s['y_sample_vs1{}'.format(_n_water)] = self.ip['energy{}'.format(_n_water)][0 : self.N_train   , 1]
                    self.t_s['y_sample_vs2{}'.format(_n_water)] = self.ip['energy{}'.format(_n_water)][0 : self.N_train   , 1]
                    
                    self.t_s['weights{}'.format(_n_water)]      = self.ip['weight{}'.format(_n_water)]/np.std(self.ip['energy{}'.format(_n_water)][: , 1])
                    self.t_s['weights_vs1{}'.format(_n_water)]  = self.ip['weight{}'.format(_n_water)]/np.std(self.ip['energy{}'.format(_n_water)][: , 1])
                    self.t_s['weights_vs2{}'.format(_n_water)]  = self.ip['weight{}'.format(_n_water)]/np.std(self.ip['energy{}'.format(_n_water)][: , 1])
                    
                    self.t_s['y_test{}'.format(_n_water)]       = self.ip['energy{}'.format(_n_water)][self.N_train : self.N_train + self.N_test , 0]
                    self.t_s['y_test_vs1{}'.format(_n_water)]   = self.ip['energy{}'.format(_n_water)][self.N_train : self.N_train + self.N_test , 0]
                    self.t_s['y_test_vs2{}'.format(_n_water)]   = self.ip['energy{}'.format(_n_water)][self.N_train : self.N_train + self.N_test , 0]
                    
                    self.t_s['x_sample{}'.format(_n_water)]     = self.ip['d_vet{}'.format(_n_water)][0 : self.N_train , :]
                    self.t_s['x_sample_vs1{}'.format(_n_water)] = self.ip['d_vet_vs1{}'.format(_n_water)][0 : self.N_train , :]
                    self.t_s['x_sample_vs2{}'.format(_n_water)] = self.ip['d_vet_vs2{}'.format(_n_water)][0 : self.N_train , :]
                    
                    self.t_s['x_test{}'.format(_n_water)]       = self.ip['d_vet{}'.format(_n_water)][self.N_train : self.N_train + self.N_test , :]
                    self.t_s['x_test_vs1{}'.format(_n_water)]   = self.ip['d_vet_vs1{}'.format(_n_water)][self.N_train : self.N_train + self.N_test , :]
                    self.t_s['x_test_vs2{}'.format(_n_water)]   = self.ip['d_vet_vs2{}'.format(_n_water)][self.N_train : self.N_train + self.N_test , :]
                    
        self.t_s['flag']         = self.flag
        return self.t_s
            
    def eval_energy(self, flag):
        """
            Function which takes as input the data of our system (output of input_setup()) 
            and returns the estimated values of the energies of each frame as Q, C6, C12
        """
        flags = ["test", "train"]
        if flag not in flags: 
            print("choose a proper flag between {}, {}".format(flags, flag))
        
        for _n_water in self.ip['N_water']:
            if flag == "test":
                
                dt     = self.t_s['x_test{}'.format(_n_water)]
                if self.ip['n_vs'] == 2:
                    dt_vs1 = self.t_s['x_test_vs1{}'.format(_n_water)]
                    dt_vs2 = self.t_s['x_test_vs2{}'.format(_n_water)]

            elif flag == "train":
                dt     = self.t_s['x_sample{}'.format(_n_water)]
                if self.ip['n_vs'] == 2:
                    dt_vs1 = self.t_s['x_sample_vs1{}'.format(_n_water)]
                    dt_vs2 = self.t_s['x_sample_vs2{}'.format(_n_water)]

            if self.ip['n_vs'] == 0:
                # Charge Contribution
                q   = k_c*np.multiply(np.power(dt[:,1:], -1), self.ip['topol{}'.format(_n_water)][0,1:])
                q   = np.sum(q,1)


                # C6 Contribution
                c6  = np.multiply(np.power(dt[:,1:], -6), self.ip['topol{}'.format(_n_water)][1,1:])
                c6  = np.sum(c6,1)

                # C12 Contribution
                c12  = np.multiply(np.power(dt[:,1:], -12), self.ip['topol{}'.format(_n_water)][2,1:])
                c12  = np.sum(c12,1)

                        
            elif self.ip['n_vs'] == 2:
                dt     = dt[:,1:-2]
                dt_vs1 = dt_vs1[:,1:-2]
                dt_vs2 = dt_vs2[:,1:-2]

                # Charge Contribution
                q   = k_c*np.multiply(np.power(dt, -1), self.ip['topol{}'.format(_n_water)][0,1:-2])
                q   = np.sum(q,1)

                q_vs1   = k_c*np.multiply(np.power(dt_vs1, -1), self.ip['topol{}'.format(_n_water)][0,1:-2])
                q_vs1   = np.sum(q_vs1,1)

                q_vs2   = k_c*np.multiply(np.power(dt_vs2, -1), self.ip['topol{}'.format(_n_water)][0,1:-2])
                q_vs2   = np.sum(q_vs2,1)

                # C6 Contribution
                c6  = np.multiply(np.power(dt, -6), self.ip['topol{}'.format(_n_water)][1,1:-2])
                c6  = np.sum(c6,1)

                c6_vs1  = np.multiply(np.power(dt_vs1, -6), self.ip['topol{}'.format(_n_water)][1,1:-2])
                c6_vs1  = np.sum(c6_vs1,1)

                c6_vs2  = np.multiply(np.power(dt_vs2, -6), self.ip['topol{}'.format(_n_water)][1,1:-2])
                c6_vs2  = np.sum(c6_vs2,1)

                # C12 Contribution
                c12  = np.multiply(np.power(dt, -12), self.ip['topol{}'.format(_n_water)][2,1:-2])
                c12  = np.sum(c12,1)

                c12_vs1  = np.multiply(np.power(dt_vs1, -12), self.ip['topol{}'.format(_n_water)][2,1:-2])
                c12_vs1  = np.sum(c12_vs1,1)

                c12_vs2  = np.multiply(np.power(dt_vs2, -12), self.ip['topol{}'.format(_n_water)][2,1:-2])
                c12_vs2  = np.sum(c12_vs2,1)

            if flag == "test":
                self.test_set['q_test{}'.format(_n_water)]   = np.array(q)
                self.test_set['c6_test{}'.format(_n_water)]  = np.array(c6)
                self.test_set['c12_test{}'.format(_n_water)] = np.array(c12)

                if self.ip['n_vs'] == 2:
                    self.test_set['q_test_vs1{}'.format(_n_water)]   = np.array(q_vs1)
                    self.test_set['c6_test_vs1{}'.format(_n_water)]  = np.array(c6_vs1)
                    self.test_set['c12_test_vs1{}'.format(_n_water)] = np.array(c12_vs1)

                    self.test_set['q_test_vs2{}'.format(_n_water)]   = np.array(q_vs2)
                    self.test_set['c6_test_vs2{}'.format(_n_water)]  = np.array(c6_vs2)
                    self.test_set['c12_test_vs2{}'.format(_n_water)] = np.array(c12_vs2)        
                    
                    self.test_set['q_test_vs{}'.format(_n_water)]   = np.array(q_vs1) + np.array(q_vs2)
                    self.test_set['c6_test_vs{}'.format(_n_water)]  = np.array(c6_vs1) + np.array(c6_vs2) 
                    self.test_set['c12_test_vs{}'.format(_n_water)] = np.array(c12_vs1) + np.array(c12_vs2)



            elif flag == "train":
                self.t_s['q_train{}'.format(_n_water)]   = np.array(q)
                self.t_s['c6_train{}'.format(_n_water)]  = np.array(c6)
                self.t_s['c12_train{}'.format(_n_water)] = np.array(c12)

                if self.ip['n_vs'] == 2:
                    self.t_s['q_train_vs1{}'.format(_n_water)]   = np.array(q_vs1)
                    self.t_s['c6_train_vs1{}'.format(_n_water)]  = np.array(c6_vs1)
                    self.t_s['c12_train_vs1{}'.format(_n_water)] = np.array(c12_vs1)

                    self.t_s['q_train_vs2{}'.format(_n_water)]   = np.array(q_vs2)
                    self.t_s['c6_train_vs2{}'.format(_n_water)]  = np.array(c6_vs2)
                    self.t_s['c12_train_vs2{}'.format(_n_water)] = np.array(c12_vs2)
                    
                    self.t_s['q_train_vs{}'.format(_n_water)]   = np.array(q_vs1) + np.array(q_vs2)
                    self.t_s['c6_train_vs{}'.format(_n_water)]  = np.array(c6_vs1) + np.array(c6_vs2) 
                    self.t_s['c12_train_vs{}'.format(_n_water)] = np.array(c12_vs1) + np.array(c12_vs2)
                    
        if flag == "train":
            return self.t_s
        elif flag == "test":
             return self.test_set
        else:
            print("not working here")
            return -1
                    
    def initialize_force(self):
        """
            Initialize force set
        """
        if self.ip['n_vs'] == 0:
            for _n_water in self.ip['N_water']:
                d, f = [], []
                for i in np.arange(np.size(self.ip['d_vet{}'.format(_n_water)],0)):
                    aux = np.array([])
                    d.append(np.append(aux, [self.ip['d_mat{}'.format(_n_water)][i,:,0] , self.ip['d_vet{}'.format(_n_water)][i,:]]))
                    aux = np.array([])
                    d.append(np.append(aux, [self.ip['d_mat{}'.format(_n_water)][i,:,1] , self.ip['d_vet{}'.format(_n_water)][i,:]]))
                    aux = np.array([])
                    d.append(np.append(aux, [self.ip['d_mat{}'.format(_n_water)][i,:,2] , self.ip['d_vet{}'.format(_n_water)][i,:]]))
                    f.append(self.ip['force{}'.format(_n_water)][i, 0 , :])


                f = np.reshape(f,[np.size(f,0)*3])           # Vector of forces applied to the ion
                d = np.array(d)
                self.t_s['x_sample{}'.format(_n_water)]     =  d[0 : self.N_train*3, :]
                self.t_s['y_sample{}'.format(_n_water)]     =  f[0 : self.N_train*3] 
                self.t_s['x_test{}'.format(_n_water)]       =  d[self.N_train*3  : self.N_train*3 + self.N_test*3, :]
                self.t_s['y_test{}'.format(_n_water)]       =  f[self.N_train*3  : self.N_train*3 + self.N_test*3] 
                self.t_s['weights{}'.format(_n_water)]      = self.ip['weight{}'.format(_n_water)]/np.std(f)
                self.t_s['force{}'.format(_n_water)]        = f

            self.t_s['flag']         = self.flag

        elif self.ip['n_vs'] == 2:
            for _n_water in self.ip['N_water']:
                d, f         = [], []
                d_vs1, f_vs1 = [], []
                d_vs2, f_vs2 = [], []
                for i in np.arange(np.size(self.ip['d_vet{}'.format(_n_water)],0)):
                    aux = np.array([])
                    d.append(np.append(aux, [self.ip['d_mat{}'.format(_n_water)][i,:,0] , self.ip['d_vet{}'.format(_n_water)][i,:]]))
                    aux = np.array([])
                    d.append(np.append(aux, [self.ip['d_mat{}'.format(_n_water)][i,:,1] , self.ip['d_vet{}'.format(_n_water)][i,:]]))
                    aux = np.array([])
                    d.append(np.append(aux, [self.ip['d_mat{}'.format(_n_water)][i,:,2] , self.ip['d_vet{}'.format(_n_water)][i,:]]))
                    f.append(self.ip['force{}'.format(_n_water)][i, 0 , :])


                    aux = np.array([])
                    d_vs1.append(np.append(aux, [self.ip['d_mat_vs1{}'.format(_n_water)][i,:,0] , self.ip['d_vet_vs1{}'.format(_n_water)][i,:]]))
                    aux = np.array([])
                    d_vs1.append(np.append(aux, [self.ip['d_mat_vs1{}'.format(_n_water)][i,:,1] , self.ip['d_vet_vs1{}'.format(_n_water)][i,:]]))
                    aux = np.array([])
                    d_vs1.append(np.append(aux, [self.ip['d_mat_vs1{}'.format(_n_water)][i,:,2] , self.ip['d_vet_vs1{}'.format(_n_water)][i,:]]))
                    f_vs1.append(self.ip['force{}'.format(_n_water)][i, 0 , :])

                    aux = np.array([])
                    d_vs2.append(np.append(aux, [self.ip['d_mat_vs2{}'.format(_n_water)][i,:,0] , self.ip['d_vet_vs2{}'.format(_n_water)][i,:]]))
                    aux = np.array([])
                    d_vs2.append(np.append(aux, [self.ip['d_mat_vs2{}'.format(_n_water)][i,:,1] , self.ip['d_vet_vs2{}'.format(_n_water)][i,:]]))
                    aux = np.array([])
                    d_vs2.append(np.append(aux, [self.ip['d_mat_vs2{}'.format(_n_water)][i,:,2] , self.ip['d_vet_vs2{}'.format(_n_water)][i,:]]))
                    f_vs2.append(self.ip['force{}'.format(_n_water)][i, 0 , :])


                f     = np.reshape(f,[np.size(f,0)*3])
                f_vs1 = np.reshape(f_vs1, [np.size(f_vs1,0)*3])
                f_vs2 = np.reshape(f_vs2,[np.size(f_vs2,0)*3]) 

                d     = np.array(d)
                d_vs1 = np.array(d_vs1)
                d_vs2 = np.array(d_vs2)

                self.t_s['x_sample{}'.format(_n_water)]     =  d[0 : self.N_train*3, :]
                self.t_s['x_sample_vs1{}'.format(_n_water)] =  d_vs1[0 : self.N_train*3, :]
                self.t_s['x_sample_vs2{}'.format(_n_water)] =  d_vs2[0 : self.N_train*3, :]

                self.t_s['y_sample{}'.format(_n_water)]     =  f[0 : self.N_train*3] 
                self.t_s['y_sample_vs1{}'.format(_n_water)] =  f_vs1[0 : self.N_train*3] 
                self.t_s['y_sample_vs2{}'.format(_n_water)] =  f_vs2[0 : self.N_train*3] 

                self.t_s['x_test{}'.format(_n_water)]       =  d[self.N_train*3  : self.N_train*3 + self.N_test*3, :]
                self.t_s['x_test_vs1{}'.format(_n_water)]   =  d_vs1[self.N_train*3  : self.N_train*3 + self.N_test*3, :]
                self.t_s['x_test_vs2{}'.format(_n_water)]   =  d_vs2[self.N_train*3  : self.N_train*3 + self.N_test*3, :]

                self.t_s['y_test{}'.format(_n_water)]       =  f[self.N_train*3  : self.N_train*3 + self.N_test*3] 
                self.t_s['y_test_vs1{}'.format(_n_water)]   =  f_vs1[self.N_train*3  : self.N_train*3 + self.N_test*3] 
                self.t_s['y_test_vs2{}'.format(_n_water)]   =  f_vs2[self.N_train*3  : self.N_train*3 + self.N_test*3] 

                self.t_s['weights{}'.format(_n_water)]      = self.ip['weight{}'.format(_n_water)]/np.std(f)
                self.t_s['weights_vs1{}'.format(_n_water)]  = self.ip['weight{}'.format(_n_water)]/np.std(f)
                self.t_s['weights_vs2{}'.format(_n_water)]  = self.ip['weight{}'.format(_n_water)]/np.std(f)

                self.t_s['force{}'.format(_n_water)]        = f
        return self.t_s      
             
    def eval_force(self,flag):
        """
            Function which takes as input the data of our system (output of input_setup())
            and returns the estimated values of the forces of each frame as Q, C6, C1
        """
        flags = ["test", "train"]
        if flag not in flags:
            print("choose a proper flag between {}, {}".format(flags, flag))

        for _n_water in self.ip['N_water']:
            if flag == "test":
                
                x     = self.t_s['x_test{}'.format(_n_water)]
                if self.ip['n_vs'] == 2:
                    x_vs1 = self.t_s['x_test_vs1{}'.format(_n_water)]
                    x_vs2 = self.t_s['x_test_vs2{}'.format(_n_water)]
                
            elif flag == "train":
                x     = self.t_s['x_sample{}'.format(_n_water)]
                if self.ip['n_vs'] == 2:
                    x_vs1 = self.t_s['x_sample_vs1{}'.format(_n_water)]
                    x_vs2 = self.t_s['x_sample_vs2{}'.format(_n_water)]

            if self.ip['n_vs'] == 0:
                N_iter = np.size(self.t_s['x_sample{}'.format(_n_water)], 1)/2
                N_iter = int(N_iter)

                dt  = x[:, N_iter : N_iter*2]
                dt  = dt[:,1:]

                dc  = x[:, 0 : N_iter ]
                dc  = dc[:,1:]
                
                # Charge Contribution
                n_q = -1
                q   = n_q*k_c*self.ip['topol{}'.format(_n_water)][0,1:]*dc[:,:]*(np.power(dt[:,:], (n_q - 2)))
                q   = np.sum(q,1)


                # C6 Contribution
                n_6 = -6
                c6  = n_6*np.multiply(dc[:,:],(np.power(dt[:,:], (n_6 - 2))))
                c6  = c6* self.ip['topol{}'.format(_n_water)][1,1:]
                c6  = np.sum(c6,1)

                # C12 Contribution
                n_12 = -12
                c12  = n_12*np.multiply(dc[:,:],np.power(dt[:,:], (n_12 - 2)))
                c12  = c12* self.ip['topol{}'.format(_n_water)][2,1:]
                c12  = np.sum(c12,1) 

            elif self.ip['n_vs'] == 2:

                N_iter = np.size(self.t_s['x_sample{}'.format(_n_water)], 1)/2
                N_iter = int(N_iter)

                dc     = x[:, 0 : N_iter ]
                dt     = x[:, N_iter : N_iter*2]
                dc     = dc[:,1:-2]
                dt     = dt[:,1:-2]
                
                dc_vs1 = x_vs1[:, 0 : N_iter ]
                dt_vs1 = x_vs1[:, N_iter : N_iter*2]
                dc_vs1 = dc_vs1[:,1:-2]
                dt_vs1 = dt_vs1[:,1:-2]
                
                dc_vs2 = x_vs2[:, 0 : N_iter ]
                dt_vs2 = x_vs2[:, N_iter : N_iter*2]
                dc_vs2 = dc_vs2[:,1:-2]
                dt_vs2 = dt_vs2[:,1:-2]

                # Charge Contribution
                n_q = -1
                q   = n_q*k_c*self.ip['topol{}'.format(_n_water)][0,1:-2]*dc[:,:]*(np.power(dt[:,:], (n_q - 2)))
                q   = np.sum(q,1)
                q_vs1   = n_q*k_c*self.ip['topol{}'.format(_n_water)][0,1:-2]*dc[:,:]*(np.power(dt_vs1[:,:], (n_q - 2)))
                q_vs1   = np.sum(q_vs1,1)
                q_vs2   = n_q*k_c*self.ip['topol{}'.format(_n_water)][0,1:-2]*dc[:,:]*(np.power(dt_vs2[:,:], (n_q - 2)))
                q_vs2   = np.sum(q_vs2,1)


                # C6 Contribution
                n_6 = -6
                c6  = n_6*np.multiply(dc[:,:],(np.power(dt[:,:], (n_6 - 2))))
                c6  = c6* self.ip['topol{}'.format(_n_water)][1,1:-2]
                c6  = np.sum(c6,1)
                c6_vs1  = n_6*np.multiply(dc_vs1[:,:],(np.power(dt_vs1[:,:], (n_6 - 2))))
                c6_vs1  = c6_vs1* self.ip['topol{}'.format(_n_water)][1,1:-2]
                c6_vs1  = np.sum(c6_vs1,1)
                c6_vs2  = n_6*np.multiply(dc_vs2[:,:],(np.power(dt_vs2[:,:], (n_6 - 2))))
                c6_vs2  = c6_vs2* self.ip['topol{}'.format(_n_water)][1,1:-2]
                c6_vs2  = np.sum(c6_vs2,1)

                # C12 Contribution
                n_12 = -12
                c12  = n_12*np.multiply(dc[:,:],np.power(dt[:,:], (n_12 - 2)))
                c12  = c12* self.ip['topol{}'.format(_n_water)][2,1:-2]
                c12  = np.sum(c12,1)    
                c12_vs1  = n_12*np.multiply(dc_vs1[:,:],np.power(dt_vs1[:,:], (n_12 - 2)))
                c12_vs1  = c12_vs1* self.ip['topol{}'.format(_n_water)][2,1:-2]
                c12_vs1  = np.sum(c12_vs1,1)
                c12_vs2  = n_12*np.multiply(dc_vs2[:,:],np.power(dt_vs2[:,:], (n_12 - 2)))
                c12_vs2  = c12_vs2* self.ip['topol{}'.format(_n_water)][2,1:-2]
                c12_vs2  = np.sum(c12_vs2,1)

            if flag == "test":
                self.test_set['q_test{}'.format(_n_water)]   = np.array(q)
                self.test_set['c6_test{}'.format(_n_water)]  = np.array(c6)
                self.test_set['c12_test{}'.format(_n_water)] = np.array(c12)

                if self.ip['n_vs'] == 2:
                    self.test_set['q_test_vs1{}'.format(_n_water)]   = np.array(q_vs1)
                    self.test_set['c6_test_vs1{}'.format(_n_water)]  = np.array(c6_vs1)
                    self.test_set['c12_test_vs1{}'.format(_n_water)] = np.array(c12_vs1)

                    self.test_set['q_test_vs2{}'.format(_n_water)]   = np.array(q_vs2)
                    self.test_set['c6_test_vs2{}'.format(_n_water)]  = np.array(c6_vs2)
                    self.test_set['c12_test_vs2{}'.format(_n_water)] = np.array(c12_vs2)          
                    
                    self.test_set['q_test_vs{}'.format(_n_water)]   = np.array(q_vs1) + np.array(q_vs2)
                    self.test_set['c6_test_vs{}'.format(_n_water)]  = np.array(c6_vs1) + np.array(c6_vs2) 
                    self.test_set['c12_test_vs{}'.format(_n_water)] = np.array(c12_vs1) + np.array(c12_vs2)


            elif flag == "train":
                self.t_s['q_train{}'.format(_n_water)]   = np.array(q)
                self.t_s['c6_train{}'.format(_n_water)]  = np.array(c6)
                self.t_s['c12_train{}'.format(_n_water)] = np.array(c12)

                if self.ip['n_vs'] == 2:
                    self.t_s['q_train_vs1{}'.format(_n_water)]   = np.array(q_vs1)
                    self.t_s['c6_train_vs1{}'.format(_n_water)]  = np.array(c6_vs1)
                    self.t_s['c12_train_vs1{}'.format(_n_water)] = np.array(c12_vs1)

                    self.t_s['q_train_vs2{}'.format(_n_water)]   = np.array(q_vs2)
                    self.t_s['c6_train_vs2{}'.format(_n_water)]  = np.array(c6_vs2)
                    self.t_s['c12_train_vs2{}'.format(_n_water)] = np.array(c12_vs2)
                    
                    self.t_s['q_train_vs{}'.format(_n_water)]   = np.array(q_vs1) + np.array(q_vs2)
                    self.t_s['c6_train_vs{}'.format(_n_water)]  = np.array(c6_vs1) + np.array(c6_vs2) 
                    self.t_s['c12_train_vs{}'.format(_n_water)] = np.array(c12_vs1) + np.array(c12_vs2)
        if flag == "train":
            return self.t_s
        elif flag == "test":
            return self.test_set
        else:
            print("not working here")
            return -1
                


class training_set():
    """
        Takes as input the energy and the force training set and build the set of function X,W,v,y
        necessary for the fitting procedure

        Input Parameters:
            energy_ts energy training set
            force_ts force training set

        Returns:
            data_set
    """
    def __init__(self, energy_ts, force_ts):
        """
            Initialization, system sensitive
        """

        self.energy_ts    = energy_ts
        self.e_ts         = energy_ts.t_s
        self.force_ts     = force_ts    
        self.f_ts         = force_ts.t_s
        self.ip = energy_ts.ip
            
    def build_weight(self):
        """
            Function which collects in one array all the weight and builds the weight matrix
        """ 
        w_vector = np.array([])
        
        # Energies
        w_e128 = self.energy_ts.t_s['weights128']
        ne_128 = np.size(self.energy_ts.t_s['y_sample128'],0)
        w_e32  = self.energy_ts.t_s['weights32']
        ne_32  = np.size(self.energy_ts.t_s['y_sample32'],0)
        # Forces
        w_f128 = self.force_ts.t_s['weights128']
        nf_128 = np.size(self.force_ts.t_s['y_sample128'],0)
        w_f32  = self.force_ts.t_s['weights32']
        nf_32  = np.size(self.force_ts.t_s['y_sample32'],0)
        
        w_vector = np.hstack([
            w_e128*np.ones(ne_128)*(1/ne_128),
            w_e32*np.ones(ne_32)*(1/ne_32),
            w_f128*np.ones(nf_128)*(1/nf_128),
            w_f32*np.ones(nf_32)*(1/nf_32)])
           
        w_vector = np.array(w_vector)
        w_matrix = np.diag(w_vector)
        return np.array(w_vector), np.matrix(w_matrix)


    def model_descriptor(self):
        """
            Method which collects the non scaled descriptor of our system
        """

        if self.energy_ts.ip['n_vs'] == 0:
            self.q   =  np.hstack([self.e_ts['q_train128'],   self.e_ts['q_train32'],  
                                   self.f_ts['q_train128'],   self.f_ts['q_train32']])
            self.c6  = -np.hstack([self.e_ts['c6_train128'],  self.e_ts['c6_train32'], 
                                   self.f_ts['c6_train128'],  self.f_ts['c6_train32']])
            self.c12 =  np.hstack([self.e_ts['c12_train128'], self.e_ts['c12_train32'],
                                   self.f_ts['c12_train128'], self.f_ts['c12_train32']])
            
            self.e_ts['y_sample128'] = np.squeeze(self.e_ts['y_sample128'])
            self.e_ts['y_sample32']  = np.squeeze(self.e_ts['y_sample32'])
            
            self.y_sample = np.hstack([self.e_ts['y_sample128'], self.e_ts['y_sample32'], 
                                       self.f_ts['y_sample128'], self.f_ts['y_sample32']])
            self.w_vector, self.w_matrix = self.build_weight()

        elif self.energy_ts.ip['n_vs'] == 2:
            self.q     = np.hstack([self.e_ts['q_train128'], self.e_ts['q_train32'], 
                                    self.f_ts['q_train128'], self.f_ts['q_train32']])
            self.q_vs1 = np.hstack([self.e_ts['q_train_vs1128'], self.e_ts['q_train_vs132'], 
                                    self.f_ts['q_train_vs1128'], self.f_ts['q_train_vs132']])
            self.q_vs2 = np.hstack([self.e_ts['q_train_vs2128'], self.e_ts['q_train_vs232'],
                                    self.f_ts['q_train_vs2128'], self.f_ts['q_train_vs232']])
            self.q_vs  = self.q_vs1 + self.q_vs2

            self.c6     = -np.hstack([self.e_ts['c6_train128'],     self.e_ts['c6_train32'],   
                                      self.f_ts['c6_train128'],     self.f_ts['c6_train32']])
            self.c6_vs1 = -np.hstack([self.e_ts['c6_train_vs1128'], self.e_ts['c6_train_vs132'],
                                      self.f_ts['c6_train_vs1128'], self.f_ts['c6_train_vs132']])
            self.c6_vs2 = -np.hstack([self.e_ts['c6_train_vs2128'], self.e_ts['c6_train_vs232'], 
                                      self.f_ts['c6_train_vs2128'], self.f_ts['c6_train_vs232']])
            self.c6_vs  = self.c6_vs1 + self.c6_vs2
            
            self.c12     = np.hstack([self.e_ts['c12_train128'],     self.e_ts['c12_train32'], 
                                      self.f_ts['c12_train128'],     self.f_ts['c12_train32']])
            self.c12_vs1 = np.hstack([self.e_ts['c12_train_vs1128'], self.e_ts['c12_train_vs132'], 
                                      self.f_ts['c12_train_vs1128'], self.f_ts['c12_train_vs132']])
            self.c12_vs2 = np.hstack([self.e_ts['c12_train_vs2128'], self.e_ts['c12_train_vs232'],
                                      self.f_ts['c12_train_vs2128'], self.f_ts['c12_train_vs232']])
            self.c12_vs  = self.c12_vs1 + self.c12_vs2
            
            self.e_ts['y_sample128'] = np.squeeze(self.e_ts['y_sample128'])
            self.e_ts['y_sample32']  = np.squeeze(self.e_ts['y_sample32'])

            self.y_sample = np.hstack([self.e_ts['y_sample128'], self.e_ts['y_sample32'], 
                                       self.f_ts['y_sample128'], self.f_ts['y_sample32']])
            self.w_vector, self.w_matrix = self.build_weight()

        
        self.borders     = self.energy_ts.ip['borders']
        self.nfunctions  = self.energy_ts.ip['n_functions']
        self.q_value     = self.energy_ts.ip['q_value']

