import numpy as np

def distance_ion(mat, n_atom, index, n_conf, n_vs = 0):
    """
        Function which evaluate the distances between our ion and the water molecules,
        returns a vector
        
        Inputs:

            Case index = 1 --> Single ion in aqueous solution
            Case index = 2 --> Single ion with virtual sites in acqueous solution
        
            mat    = matrix of coordinates
            n_atom = number of atoms of the system
            index  = index of or the single ion, or the last virtual site 
            n_vs   = number of virtual sites

        Returns:
           np array of Euclidian distances
    """
    
    if index == 1:
        d      = []
        for i in np.arange(n_conf):
            d.append(np.array([np.linalg.norm(mat[int(i),0,:] - dis) for dis in mat[int(i),:,:]]))
        return np.array(d, dtype='float')
    
    elif index != 1:
        #Case of Virtual Site
        if n_vs == 2:
            n_atom = n_atom + n_vs
            d, d_vs1, d_vs2      = [], [], []
            
            # For each configurations evaluate a vector distance ---> Euclidian distance (linalg.norm)
            for i in np.arange(n_conf):
                d_aux, d_vs1_aux, d_vs2_aux = [], [], []
                
                for dis in mat[int(i),:,:]:
                    d_aux.append( np.linalg.norm(mat[int(i),0,:] - dis) )
                    
                    d_vs1_aux.append(np.linalg.norm(mat[int(i),-1,:] - dis))
                    d_vs2_aux.append(np.linalg.norm(mat[int(i),-2,:] - dis))
    
                d.append(d_aux)
                d_vs1.append(d_vs1_aux)
                d_vs2.append(d_vs2_aux)
                
            return np.array(d, dtype='float'), np.array(d_vs1, dtype='float'), np.array(d_vs2, dtype='float')
            

def distance_ion_mat(mat, n_atom, index, n_conf, n_vs = 0):
    """
        Function which evaluate the distances between our ion and the water molecules,
        returns a matrix
        
        Inputs:
            Case index = 1 --> Single ion in aqueous solution
            Case index = 2 --> SIngle ion with virtual sites in acqueous solution
            
            mat    = matrix of coordinates
            n_atom = number of atoms of the system
            index  = index of or the single ion, or the last virtual site 
            n_vs   = number of virtual sites
        
        Returns:
            np array of matrix deviation distance

    """
    if index == 1:
        d      = []
        for i in np.arange(n_conf):
            b  = np.ones([n_atom,3])*mat[int(i), 0, :]
            c  = mat[int(i), :, :] - b
            d.append(c)
        d  = np.array(d)
        return d
    elif index != 1:
               
        if n_vs == 2:
            #    Linear Geometry
            n_atom = n_atom + n_vs
            d, dvs1, dvs2      = [], [], []
            
            # For each configurations evaluate a vector matrix ---> Difference
            for i in np.arange(n_conf):
                b     = np.ones([n_atom,3])*mat[int(i), 0, :]
                bvs1  = np.ones([n_atom,3])*mat[int(i), -1, :]
                bvs2  = np.ones([n_atom,3])*mat[int(i), -2, :]
                
                c     = mat[int(i), :, :] - b 
                cvs1  = mat[int(i), :, :] - bvs1
                cvs2  = mat[int(i), :, :] - bvs2
                
                d.append(c)
                dvs1.append(cvs1)
                dvs2.append(cvs2)
                
            return np.array(d), np.array(dvs1), np.array(dvs2)
