import sys
import os
import cvxpy as cp
import numpy as np
from ControlRF import GPController, KernelADPGP, RandomFeaturesADPGP

module_path = os.path.abspath(os.path.join('.'))
os.environ['MOSEKLM_LICENSE_FILE'] = module_path
import mosek
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path + "/core")


class ADPController(GPController):
    def __init__(self, system_est, system, controller, aff_lyap, x_0, T, num_steps, estimate, rf_d=50):
        xs, ys, zs = GPController.__init__(self, system_est, system, controller, aff_lyap, x_0, T, num_steps)
        self.gp = ADPController._train_gp(xs, ys, zs, estimate, rf_d)
        print("----------------------------training finished------------------------------")

    
    @staticmethod
    def _train_gp(xs, ys, zs, estimate, rf_d):
        '''training for GP'''
        if estimate:
            gp = RandomFeaturesADPGP(xs, ys, zs, rf_d)
            gp.train()
            gp.c_train()
        else:
            gp = KernelADPGP(xs, ys, zs)
            gp.train()
            gp.c_train()        
        return gp