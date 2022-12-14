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
        self.inv_kernel = []
        self.k_vec = []

    def _compute_kernel_helper(self, x_test, test=False):
        train_k = np.exp(-np.sum(np.square(self.x_train), axis=2, keepdims=True)/(2 * self.sigma ** 2)) #(n,1)
        if test:
            test_k = np.exp(-np.sum(np.square(x_test), axis=2, keepdims=True)/(2 * self.sigma ** 2)) #(n_t,1)
        else:
            test_k = train_k
        k_mat = train_k @ test_k.T #(n,n_t)
        x_dif = self.x_train.reshape((self.n,1,self.d)) - x_test #(n,n_t,d)
        diag_k = np.exp(-np.sum(np.square(x_dif), axis=2)/(2 * self.sigma ** 2)) #(n,n_t)
        return k_mat, diag_k

    def _compute_kernel(self, k_mat, diag_k, y_test):
        ys = np.sum(self.y_train, 1, keepdims=True) @ np.sum(y_test, 1) #(n,n_t)
        diag_diff = (self.y_train @ y_test.T) * (diag_k - k_mat) #(n,n_t)
        return k_mat * ys + diag_diff #(n,n_t)

    def _compute_entries(self, x_test, y_test):
        ys = np.square(np.sum(y_test,1)) #(n_t)
        test_k = np.exp(-np.sum(np.square(x_test), axis=2)/(2 * self.sigma ** 2)) #(n_t)
        k_diag = np.einsum('ij,ji->i',test_k, test_k.T) #(n_t)
        diff = np.einsum('ij,ji->i', y_test, y_test.T) * (1 - k_diag) #(n_t)
        return k_diag * ys + diff #(n_t)

    def train(self): 
        # kernel training
        self.kernel = self._compute_kernel(self.x_train, self.y_train)
        self.inv_kernel = la.inv(self.kernel + self.sigma_n**2 * np.identity(self.n)) #(n,n)          

    def test(self, x_test, y_test):
        # c = (K+sigma^2I)^{-1}K(x)
        self.n_t = len(x_test)
        self.k_vec = self._compute_kernel(x_test, y_test, test=True) #(n,n_t)
        return  self.z_train @ self.inv_kernel @ self.k_vec #n_t
    
    def sigma(self, x_test, y_test): #n_t=1
        return self._compute_entries(x_test, y_test) - \
            np.einsum('ij,jk,ki->i', self.k_vec.T, self.inv_ckernel, self.k_vec)
    
    def _compute_k_h(self, x_test): #n_t=1
        k_mat, diag_k = self._compute_kernel_helper(x_test, test=True) #(n,n_t)
        diag_diff = (diag_k- k_mat) * self.y_train #(n,m+1)
        self.k_h = k_mat * np.sum(self.y_train, 1, keepdims=True) + diag_diff #n,n_t * n,1?????
        return


    def mean_var(self, x_test): #n_t=1
        self._compute_k_h(x_test)
        meanvar = self.z_train @ self.inv_ckernel @ self.k_h #m+1
        return meanvar.T #y @ meanvar
    
    def sigma_var(self): #n_t=1
        sigmavar = sqrtm(np.identity(self.m+1)-self.k_h.T@self.inv_ckernel@self.k_h) #(m+1,m+1)
        # norm(y @ sigmavar) 
        return sigmavar


