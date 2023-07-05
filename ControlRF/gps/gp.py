import numpy as np

class GaussianProcess():
    '''
    methods to overwrite:
    train, test, sigma, mean_var, sigma_var
    '''
    def __init__(self, x_train, y_train, z_train):
        self.x_train = x_train
        self.y_train = y_train
        self.z_train = z_train       
        self.n = len(x_train) #number of samples
        self.d = len(x_train[0]) #dimension of the data
        self.m = len(y_train[0]) - 1 #number of controls
        
        self.mu = 0
        self.sgm = 20 #regularization parameter
        self.rf_mu = np.zeros(self.d)
        self.rf_cov = (self.sgm ** 2) * np.identity(self.d) #gives fourier transform of the RBF kernel



