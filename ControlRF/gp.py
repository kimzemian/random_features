import numpy as np

class GaussianProcess():
    def __init__(self, x_train, y_train, z_train):
        self.x_train = x_train
        self.y_train = y_train
        self.z_train = z_train       
        self.n = len(x_train) #number of samples
        self.d = len(x_train[0]) #dimension of the data
        self.m = len(y_train[0]) - 1 #number of controls
        
        self.mu = 0
        self.sigma = 1
        self.rf_mu = np.zeros(self.d)
        self.rf_cov = 1/self.sigma * np.identity(self.d)



