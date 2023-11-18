import numpy as np
from fast_control.gp_factory import GPFactory


class GaussianProcess(GPFactory):
    """
    Gaussian Process kernel.

    methods to overwrite:
    train, test, sigma, mean_var, sigma_var
    """

    def __init__(self, x_train, y_train, z_train):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.z_train = z_train
        self.n = len(x_train)  # number of samples
        self.d = len(x_train[0])  # dimension of the data
        self.m = len(y_train[0]) - 1  # dimension of the control input
        self.mu = 0
        self.rf_mu = np.zeros(self.d)
        self.rf_cov = np.identity(self.d)  # gives fourier transform of the RBF kernel
