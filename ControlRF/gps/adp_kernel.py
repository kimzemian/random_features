import timeit
import math
import numpy as np
from numpy import linalg as la
from scipy.linalg import sqrtm
from .gp import GaussianProcess

class ADPKernel(GaussianProcess):
    '''affine dot product kernel gp'''

    __name__ = 'adp_kernel'

    def __init__(self, x_train, y_train, z_train):
        GaussianProcess.__init__(self, x_train, y_train, z_train)

    def _compute_kernel(self, x_test): 
        x_dif = self.x_train.reshape((self.n,1,self.d)) - x_test
        return np.exp(-np.sum(np.square(x_dif), axis=2)/(2 * self.sgm ** 2)) #(n,n) or (n,n_t)

    def train(self):
        tic = timeit.default_timer()
        kernel = self._compute_kernel(self.x_train)
        self.c_kernel = self.y_train @ self.y_train.T * kernel
        self.inv_ckernel = la.inv(self.c_kernel + self.sgm**2 * np.identity(self.n)) #(n,n)
        toc = timeit.default_timer()
        self.training_time = toc-tic

    def test(self,x_test=None, y_test=None):
        tic = timeit.default_timer()
        k_vec = self._compute_kernel(x_test) #(n,n_t)
        self.c_vec = self.y_train @ y_test.T * k_vec #(n,n_t)
        pred = self.z_train @ self.inv_ckernel @ self.c_vec #n_t
        toc = timeit.default_timer()
        self.test_time = toc - tic
        return pred
    
    def sigma(self, x_test=None, y_test=None): #n_t
        return np.einsum('ij,ji->i',y_test,y_test.T) - \
            np.einsum('ij,jk,ki->i', self.c_vec.T, self.inv_ckernel, self.c_vec)

    def mean_var(self, x_test): #n_t=1
        k_vec = self._compute_kernel(x_test) #(n,n_t)
        self.k_h = k_vec * self.y_train  #(n,m+1)
        meanvar = self.z_train @ self.inv_ckernel @ self.k_h #m+1
        return meanvar.T #y @ meanvar
    
    def sigma_var(self): #n_t=1
        sigmavar = sqrtm(np.identity(self.m+1)-self.k_h.T@self.inv_ckernel@self.k_h) #(m+1,m+1)
        # norm(y @ sigmavar) 
        return sigmavar


