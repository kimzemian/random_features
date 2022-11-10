from gp import *

class RandomFeaturesGP(GaussianProcess):
    #gaussian process using random features 
    def __init__(self, x_train, y_train, z_train):    
        super().__init__(x_train, y_train, z_train)
        self.rf_d = 700 #rf_d is dim of randomfeatures vector, to choose based on paper     
        self.sigma_n = 1 #regularization parameter
        self.samples = np.random.multivariate_normal\
                       (self.rf_mu,self.rf_cov,size =self.rf_d//2) #(rf_d//2,d)
        self.phi = []
        self.cphi = []
        self.phi_test = []
        self.cphi_test = []
        self.inv_phi = []
        self.inv_cphi = []

    def _compute_phi(self, x): #(n,d)  first var: n or n_t
        phi = np.empty((len(x),self.rf_d))
        dot_product = x @ self.samples.T #(n,rf_d//2)
        phi[:,0::2]=np.sin(dot_product)
        phi[:,1::2]=np.cos(dot_product)
        phi = math.sqrt(2/self.rf_d) * phi #(n,rf_d)
        return phi

    def _compute_cphi(self, phi, y): #(n,d)(n,m+1) first,third var: n or n_t
        return np.matmul(y[:,:,np.newaxis], phi[:,np.newaxis,:]).reshape(len(phi),-1) #(n,s) 

    def train(self):
        tic = timeit.default_timer()
        self.phi = self._compute_phi(self.x_train) #(n,s)
        self.inv_phi = la.inv(self.phi.T @self.phi + self.sigma_n ** 2 * \
                              np.identity(self.rf_d)) #(s,s) 
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
        pred = self.phi_test @ self.inv_phi @ self.phi.T @ self.z_train #(n_t,1)
        toc = timeit.default_timer()
        self.test_time = toc - tic
        return pred

    def c_test(self, y_test):
        self.cphi_test = self._compute_cphi(self.phi_test, y_test) #(n_t,s)
        return self.cphi_test @ self.inv_cphi @ self.cphi.T @ self.z_train #(n_t,1)

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

    def meanvar(self):
        meanvar = self.phi_test @ np.reshape(self.inv_cphi@self.cphi.T@self.z_train, self.d, -1) #(n_t,m+1)
        #y[:,:,np.newaxis] @  meanvar[:,np.newaxis,:]
    
    def sigmavar(self):
        sigmavar = sigma**2 * np.einsum('ij,jk,ki->i', self.cphi_test, np.reshape(self.inv_cphi, self.d, self.m + 1, self.m +1, self.d), self.cphi_test.T)
        y[:,:,np.newaxis] @ y[:,np.newaxis,:] * sigmavar
        #self.sigma**2 * y @ sigmavar @ y
                  