from sklearn.model_selection import train_test_split

class gen_data_set():
    """
    Class which generatesthe training and test set exploiting
    the function train_test_split from sklearn.model_selection.

    Inputs:
        input_params: dictionary of values

    Returns:
        self.x_train, self.x_test, self.y_train, self.y_test
    """
    
    def __init__(self, input_params):
        self.ip       = input_params
        self.y        = self.ip['y']
        self.H        = self.ip['x']
        self.borders  = self.ip['borders']
    
    def gen_ts(self):
        """
            Function which splits the input data in test set and training set
        """
        perc_ts = self.ip['n_confs']/self.ip['n_train'] - 1
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.H, self.y, test_size=perc_ts, random_state=42)
        
        
        
        
        
    
