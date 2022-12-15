import sys
import os
import cvxpy as cp
import numpy as np
from ControlRF import GPController, KernelGP, RandomFeaturesGP

module_path = os.path.abspath(os.path.join('.'))
os.environ['MOSEKLM_LICENSE_FILE'] = module_path
import mosek
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path + "/core")

from core.controllers import QPController

class ADController(GPController):
    def __init__(self, system_est, system, controller, aff_lyap, x_0, T, num_steps, estimate, rf_d=50):
        xs, ys, zs = GPController.__init__(self, system_est, system, controller, aff_lyap, x_0, T, num_steps)
        self.gp = ADController._train_gp(xs, ys, zs, estimate, rf_d)
        print("----------------------------training finished------------------------------")
    
    @staticmethod
    def _train_gp(xs, ys, zs, estimate, rf_d):
        '''training for GP'''
        if estimate:
            gp = RandomFeaturesGP(xs, ys, zs, rf_d)
            gp.train()
        else:
            gp = KernelGP(xs, ys, zs)
            gp.train()       
        return gp