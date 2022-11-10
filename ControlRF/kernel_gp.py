import timeit
import math
import numpy as np
from numpy import linalg as la
from scipy.linalg import sqrtm
from .gp import GaussianProcess

class KernelGP(GaussianProcess):
    def __init__(self, x_train, y_train, z_train):
        GaussianProcess.__init__(self, x_train, y_train, z_train)
        self.sigma_n = 1
        self.kernel = []
        self.c_kernel = []
        self.inv_kernel = []
        self.inv_ckernel = []
        self.k_vec = []
        self.c_vec = []

    def _compute_kernel(self, x_test): 
        x_dif = self.x_train.reshape((self.n,1,self.d)) - x_test
        return np.exp(-np.sum(np.square(x_dif), axis=2)/(2 * self.sigma ** 2)) #(n,n) or (n,n_t)

    def train(self): 
        # kernel training
        tic = timeit.default_timer()
        self.kernel = self._compute_kernel(self.x_train)
        self.inv_kernel = la.inv(self.kernel + self.sigma_n**2 * np.identity(self.n)) #(n,n)      
        toc = timeit.default_timer()
        self.training_time = toc-tic
        
    def c_train(self):
        # compound kernel training
        self.c_kernel = self.y_train @ self.y_train.T * self.kernel
        self.inv_ckernel = la.inv(self.c_kernel + self.sigma_n**2 * np.identity(self.n)) #(n,n)

    def test(self, x_test):
        # c = (K+sigma^2I)^{-1}K(x)
        self.n_t = len(x_test)
        tic = timeit.default_timer()
        self.k_vec = self._compute_kernel(x_test) #(n,n_t)
        pred = self.z_train @ self.inv_kernel @ self.k_vec #n_t
        toc = timeit.default_timer()
        self.test_time = toc - tic
        return pred

    def c_test(self, y_test):
        self.c_vec = self.y_train @ y_test.T * self.k_vec #(n,n_t)
        return  self.z_train @ self.inv_ckernel @ self.c_vec #n_t

    def sigma(self): #n_t
        #c_kernel sigma computation
        return np.ones(self.n_t) - \
            np.einsum('ij,jk,ki->i', self.k_vec.T, self.inv_kernel, self.k_vec)
    
    def c_sigma(self, y_test): #n_t
        return np.einsum('ij,ji->i',y_test,y_test.T) - \
            np.einsum('ij,jk,ki->i', self.c_vec.T, self.inv_ckernel, self.c_vec)

    def mean_var(self, x_test): #n_t=1
        self.k_vec = self._compute_kernel(x_test) #(n,n_t)
        self.k_h = self.k_vec * self.y_train  #(n,m+1)
        b = self.z_train @ self.inv_ckernel @ self.k_h #m+1
        #y @ b.T
        return b
    def sigma_var(self): #n_t=1
        c = sqrtm(np.identity(self.m+1)-self.k_h.T@self.inv_ckernel@self.k_h) #(m+1,m+1)
        # norm(y @ c) 
        return c


