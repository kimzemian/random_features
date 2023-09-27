import numpy as np
import toml


class GaussianProcess:
    """Gaussian Process kernel.

    methods to overwrite:
    train, test, sigma, mean_var, sigma_var
    """

    def __init__(self, x_train, y_train, z_train):
        with open("config.toml") as f:
            config = toml.load(f)
        sys_conf = (
            config["inverted_pendulum"] if len(y_train[0]) == 2 else config["acrobat"]
        )
        self.x_train = x_train
        self.y_train = y_train
        self.z_train = z_train
        self.n = len(x_train)  # number of samples
        self.d = len(x_train[0])  # dimension of the data
        self.m = sys_conf["m"]  # number of controls

        self.mu = 0
        self.sgm = sys_conf["sgm"]  # regularization parameter
        self.rf_mu = np.zeros(self.d)
        self.rf_cov = (self.sgm**2) * np.identity(
            self.d
        )  # gives fourier transform of the RBF kernel
