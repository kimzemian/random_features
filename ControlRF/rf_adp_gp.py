import timeit
import math
import numpy as np
from numpy import linalg as la
from scipy.linalg import sqrtm
from .gp import GaussianProcess


class RandomFeaturesADPGP(GaussianProcess):
    #gaussian process using random features 
    def __init__(self, x_train, y_train, z_train, rf_d=50):    
        super().__init__(x_train, y_train, z_train)
        self.rf_d = rf_d #rf_d is dim of randomfeatures vector, to choose based on paper     
        self.sigma_n = 1 #regularization parameter
        self.s = (self.m+1) * self.rf_d
        self.samples = np.random.multivariate_normal(self.rf_mu,self.rf_cov, \
                        size =((self.m+1)*self.rf_d//2)) #(s/2,d)
        self.phi = []
        self.cphi = []
        self.phi_test = []
        self.cphi_test = []
        self.inv_phi = []
        self.inv_cphi = []

    def _compute_phi(self, x): #(n,s)  first var: n or n_t
        phi = np.empty((len(x),self.s))
        dot_product = x @ self.samples.T #(n,s/2)
        phi[:,0::2]=np.sin(dot_product)
        phi[:,1::2]=np.cos(dot_product)
        phi = math.sqrt(2/self.rf_d) * phi
        return phi

    def _compute_cphi(self, phi, y): #(n,s) first,third var: n or n_t
        pre_cphi = y[:,:,np.newaxis] * phi.reshape((len(y),self.m+1,-1)) #(n,s) 
        return pre_cphi.reshape((len(y),-1))

    def train(self):
        tic = timeit.default_timer()
        self.phi = self._compute_phi(self.x_train) #(n,s)
        self.inv_phi = la.inv(self.phi.T @self.phi + self.sigma_n ** 2 * \
                              np.identity(self.s)) #(s,s) 
        toc = timeit.default_timer()
        self.training_time = toc-tic
    
    def c_train(self):
        self.cphi = self._compute_cphi(self.phi, self.y_train) #(n,s)
        self.inv_cphi = la.inv(self.cphi.T @self.cphi + self.sigma_n ** 2 * \
                             np.identity((self.m+1)*self.rf_d)) #(s,s)

    def test(self, x_test):
        # pred = phi(x)(Z'Z+simga^2I)^{-1}Z'y
        tic = timeit.default_timer()
        self.phi_test = self._compute_phi(x_test) #(n_t,rf_d)
        pred = self.phi_test @ self.inv_phi @ self.phi.T @ self.z_train #n_t
        toc = timeit.default_timer()
        self.test_time = toc - tic
        return pred

    def c_test(self, y_test):
        self.cphi_test = self._compute_cphi(self.phi_test, y_test) #(n_t,s)
        return self.cphi_test @ self.inv_cphi @ self.cphi.T @ self.z_train #n_t

    def sigma(self):
        #random features variance computation
        return self.sigma**2 * np.einsum('ij,jk,ki->i', self.phi_test, \
                                         self.inv_phi, self.phi_test.T)

    def c_sigma(self):
        return self.sigma**2 * np.einsum('ij,jk,ki->i', self.cphi_test, \
                                         self.inv_cphi, self.cphi_test.T)

    def estimate_kernel(self):
        return self.phi @ self.phi.T
    
    def estimate_ckernel(self):
        return self.cphi @ self.cphi.T

    def mean_var(self,x_test): #n_t=1
        self.phi_test = self._compute_phi(x_test) #(n_t,s)
        rest = np.reshape(self.inv_cphi @ self.cphi.T @ self.z_train, (self.rf_d,-1)) #(rf_d,m+1)
        meanvar = np.einsum('ij,ji->i', self.phi_test.reshape((self.m+1,-1)), rest) #(m+1)
        #y @  meanvar
        return meanvar    
            
    def sigma_var(self): #n_t=1
        test = self.phi_test.reshape((self.m+1,-1)) #(m+1,rf_d)
        inv = self.inv_cphi.reshape((-1,self.rf_d,self.m+1,self.rf_d)) #(m+1,rf_d,m+1,rf_d) 
        sigmavar = sqrtm(np.einsum('ij,ijkl,kl->ik',test,inv,test)) #(m+1,m+1)
        #norm(y @ sigmavar.T)
        return sigmavar.T
                  