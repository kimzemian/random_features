import timeit
import numpy as np
from numpy import linalg as la
from scipy.linalg import sqrtm
from .gp import GaussianProcess

class VanillaKernel(GaussianProcess):
    def __init__(self, x_train, y_train, z_train, sgm=10):
        GaussianProcess.__init__(self, x_train, y_train, z_train, sgm)

    def _compute_kernel(self, x_test): 
        x_dif = self.x_train.reshape((self.n,1,self.d)) - x_test
        return np.exp(-np.sum(np.square(x_dif), axis=2)/(2 * self.sgm ** 2)) #(n,n) or (n,n_t)

    def train(self): 
        # kernel training
        tic = timeit.default_timer()
        self.kernel = self._compute_kernel(self.x_train)
        self.inv_kernel = la.inv(self.kernel + self.sgm**2 * np.identity(self.n)) #(n,n)  
        toc = timeit.default_timer()
        self.training_time = toc-tic         

    def test(self, x_test):
        # c = (K+sgm^2I)^{-1}K(x)
        tic = timeit.default_timer()
        self.n_t = len(x_test)
        self.k_vec = self._compute_kernel(x_test) #(n,n_t)
        pred = self.z_train @ self.inv_kernel @ self.k_vec #n_t
        toc = timeit.default_timer()
        self.test_time = toc - tic
        return pred

    def sigma(self): #n_t
        #c_kernel sigma computation
        return np.ones(self.n_t) - \
            np.einsum('ij,jk,ki->i', self.k_vec.T, self.inv_kernel, self.k_vec)


