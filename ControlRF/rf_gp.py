import timeit
import math
import numpy as np
from numpy import linalg as la
from scipy.linalg import sqrtm
from .gp import GaussianProcess

class RandomFeaturesGP(GaussianProcess):
    #gaussian process using random features 
    def __init__(self, x_train, y_train, z_train,rf_d=50):    
        super().__init__(x_train, y_train, z_train)
        self.rf_d = rf_d #rf_d is dim of randomfeatures vector, to choose based on paper     
        self.sigma_n = 1 #regularization parameter
        self.samples = np.random.multivariate_normal(self.rf_mu,self.rf_cov, \
                        size =((self.m+1)*self.rf_d//2)) #(s/2,d)

    def _prephi(self, x): #(n,rf_d,m+1)  first var: n or n_t
        phi = np.empty((len(x),self.rf_d,self.m+1))
        dot_product = np.reshape(x @ self.samples.T, (len(x), self.rf_d//2, -1))  #(n,rf_d//2,m+1)
        phi[:,0::2,:]=np.sin(dot_product)
        phi[:,1::2,:]=np.cos(dot_product)
        phi = math.sqrt(2/self.rf_d) * phi
        return phi

    def _compute_cphi(self, phi, y): #(n,rf_d)  first var: n or n_t
        return np.squeeze(phi @ y[:,:,np.newaxis])

    def train(self):
        tic = timeit.default_timer()
        prephi = self._prephi(self.x_train) #(n,rf_d,m+1)
        self.phi = self._compute_cphi(prephi,self.y_train) #(n,rf_d)
        self.inv_phi = la.inv(self.phi.T @ self.phi + self.sigma_n ** 2 * \
                              np.identity(self.rf_d)) #(rf_d,rf_d) 
        toc = timeit.default_timer()
        self.training_time = toc-tic
    
    def test(self, x_test, y_test):
        # pred = phi(x)(Z'Z+simga^2I)^{-1}Z'y
        tic = timeit.default_timer()
        #self.prephi_test = self._prephi(x_test) #(n_t,rf_d,m+1)
        self.phi_test = self._compute_cphi(self.prephi_test, y_test) #(n_t,rf_d)
        pred = self.phi_test @ self.inv_phi @ self.phi.T @ self.z_train #n_t
        toc = timeit.default_timer()
        self.test_time = toc - tic
        return pred

    def csigma(self):
        #random features variance computation
        return self.sigma**2 * np.einsum('ij,jk,ki->i', self.phi_test, \
                                         self.inv_phi, self.phi_test.T)

    def estimate_kernel(self):
        return self.phi @ self.phi.T

    def mean_var(self,x_test): #n_t=1
        self.prephi_test = self._prephi(x_test).squeeze(axis=0) #(rf_d,m+1)
        meanvar = self.prephi_test.T @  self.inv_phi @ self.phi.T @ self.z_train #(n_t,1)
        #y @ meanvar 
        return meanvar
    
    def sigma_var(self): #n_t=1
        sigmavar = self.sigma * sqrtm(self.prephi_test.T @ self.inv_phi @ self.prephi_test) #(m+1,m+1)
        #norm(y @ sigmavar.T)
        return sigmavar.T         