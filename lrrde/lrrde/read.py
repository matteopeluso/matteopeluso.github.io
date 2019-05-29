from lrr_de_constant import *
from distances import *

def read_topol(filename):
    """
        function which reads a topology file and returns 
        a multi dimensional array converting the sigma/eps
        in sqrt of c6 and c12
    """
    with open(filename,'r') as f:
            aux = []
            for line in f:
                
                if len(line.split()) == 6 and 'INDEX' not in line:
                    aux.append([float(line.split()[3]), float(line.split()[4]), float(line.split()[5])])
            aux      =  np.array(aux)
            q        =  aux[:,0]
            c6       =  np.sqrt(4*aux[:,2]*aux[:,1]**6)  # Sqrt of C6
            c12      =  np.sqrt(4*aux[:,2]*aux[:,1]**12) # Sqrt of C12 
           
            topology =  np.vstack([q,c6,c12])

    return topology

def read_ener(filename, counterpoise = True):
    """
        function which reads an energy file and returns a multi
        dimensional array.
        If counterpoise True the second column is the correct one
    """
    with open(filename,'r') as f:
   
            energy = []
            if counterpoise:
                for line in f:
                    if len(line.split()) == 2 and 'raw' not in line:
                        energy.append([float(line.split()[0]), \
                                       float(line.split()[1])])
            else:
                for line in f:
                    if len(line.split()) == 2 and 'raw' not in line:
                        energy.append([float(line.split()[0])])
    energy = np.array(energy)*conv_kcal_kj   # Conversion from kcal/mol to kj/mol
    f.close()
    return energy

def read_coord(filename, n_atom, n_vs = 0):
    """
        function which reads a coordinate file, if
        the number of virtual site is null evaluate also
        the distance vector and matrix
    """
    with open(filename,'r') as f:
            coordinates = []
            for line in f:
                if len(line.split()) == 4 and 'coo' not in line:
                    coordinates.append([float(line.split()[1]), \
                                        float(line.split()[2]), \
                                        float(line.split()[3])])
            coordinates = np.array(coordinates)*0.1   # From A to nm
            n_conf = np.size(coordinates)/(n_atom*3)
            coordinates = np.reshape(coordinates,(int(n_conf),n_atom,3))
    f.close()
    return coordinates

def read_forces(filename, n_atom):
    """
        function which reads a force file
    """
    with open(filename,'r') as f:

        forces = []
        for line in f:
            if len(line.split()) == 4 and 'forces' not in line:
                forces.append([float(line.split()[1]), \
                               float(line.split()[2]), \
                               float(line.split()[3])])
        forces = np.array(forces)
        n_conf = np.size(forces)/(n_atom*3)
        forces = np.reshape(forces,(int(n_conf),n_atom,3))*conv_force                    # Conversion forces n kJ/mol nm
    f.close()
    return forces
    