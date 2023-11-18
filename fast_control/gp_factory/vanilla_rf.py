import math
import timeit

import numpy as np
from numpy import linalg as la

from .gp import GaussianProcess


class VanillaRandomFeatures(GaussianProcess):
    """Vanilla random features GP."""

    def __init__(self, x_train, y_train, z_train, sgm=10, rf_d=50):
        super().__init__(x_train, y_train, z_train, sgm)
        self.rf_d = (
            rf_d  # rf_d is dim of randomfeatures vector, to choose based on paper
        )
        self.s = (self.m + 1) * self.rf_d
        self.samples = np.random.multivariate_normal(
            self.rf_mu, self.rf_cov, size=((self.m + 1) * self.rf_d // 2)
        )  # (s/2,d)

    def _compute_phi(self, x):  # (n,s)  first var: n or n_t
        phi = np.empty((len(x), self.s))
        dot_product = x @ self.samples.T  # (n,s/2)
        phi[:, 0::2] = np.sin(dot_product)
        phi[:, 1::2] = np.cos(dot_product)
        phi = math.sqrt(2 / self.rf_d) * phi
        return phi

    def train(self):
        tic = timeit.default_timer()
        self.phi = self._compute_phi(self.x_train)  # (n,s)
        self.inv_phi = la.inv(
            self.phi.T @ self.phi + self.sgm**2 * np.identity(self.s)
        )  # (s,s)
        toc = timeit.default_timer()
        self.training_time = toc - tic

    def test(self, x_test):
        # pred = phi(x)(Z'Z+simga^2I)^{-1}Z'y
        tic = timeit.default_timer()
        self.phi_test = self._compute_phi(x_test)  # (n_t,rf_d)
        pred = self.phi_test @ self.inv_phi @ self.phi.T @ self.z_train  # n_t
        toc = timeit.default_timer()
        self.test_time = toc - tic
        return pred

    def sigma(self):
        # random features variance computation
        return self.sgm**2 * np.einsum(
            "ij,jk,ki->i", self.phi_test, self.inv_phi, self.phi_test.T
        )

    def estimate_kernel(self):
        return self.phi @ self.phi.T
