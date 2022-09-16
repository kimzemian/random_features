from gp import *

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
        self.k_vec = self._compute_kernel(x_test)
        pred = self.z_train @ self.inv_kernel @ self.k_vec #(n,n_t)
        toc = timeit.default_timer()
        self.test_time = toc - tic
        return pred

    def c_test(self, y_test):
        self.c_vec = y_test @ y_test.T * self.k_vec
        return  self.z_train @ self.inv_ckernel @ self.c_vec #(n,n_t)

    def sigma(self):
        #c_kernel sigma computation
        return np.id(self.n_t) -np.einsum('ij,jk,ki->i', self.k_vec.T, self.inv_kernel, self.k_vec)
    
    def c_sigma(self, y_test):
        return y_test@y_test.T - np.einsum('ij,jk,ki->i', self.c_vec.T, self.inv_ckernel, self.c_vec)




