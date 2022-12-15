import sys
import os
import cvxpy as cp
import numpy as np
from ControlRF import KernelADPGP, RandomFeaturesADPGP

module_path = os.path.abspath(os.path.join('.'))
os.environ['MOSEKLM_LICENSE_FILE'] = module_path
import mosek
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path + "/core")

from core.controllers import QPController

class GPController(QPController):
    def __init__(self, system_est, system, controller, aff_lyap, x_0, T, num_steps):
        QPController.__init__(self, system_est, system_est.m)
        xs, us, ts = GPController._simulate(system, controller, x_0, T, num_steps)
        xs, ys, zs = GPController._build_ccf_data(aff_lyap,xs,us,ts) #ts=ts[1:-1]
        return xs, ys, zs
    
    @staticmethod
    def _simulate(system, controller, x_0, T, num_steps):
        '''simulate system with specified controller'''
        ts = np.linspace(0, T, num_steps)
        xs, us = system.simulate(x_0, controller, ts)
        return xs, us, ts

    @staticmethod
    def _build_ccf_data(aff_lyap, xs, us, ts):
        ''' estimate error in the derivate of the CCF function
        using forward differencing '''
        av_x = (xs[:-1] + xs[1:])/2
        zs= [(aff_lyap.eval(xs[i+1],ts[i+1])- aff_lyap.eval(xs[i],ts[i]))/(ts[i+1]-ts[i]) - \
        aff_lyap.eval_dot(av_x[i],us[i],ts[i]) for i in range(len(us))]
        ys = np.concatenate((np.ones((len(us),1)), us), axis=1)
        return av_x, ys, zs
    
    def add_stability_constraint(self,aff_lyap, comp=0, slacked=False, beta=1, coeff=0):
        if slacked:
            delta = cp.Variable()
            self.variables.append(delta)
            self.static_costs.append(coeff * cp.square(delta))
            constraint = lambda x,t :  self._build_cons(x, t, aff_lyap, comp, beta, delta)
        else:
            constraint = lambda x,t :  self._build_cons(x, t, aff_lyap, comp, beta)
        self.constraints.append(constraint)

    def _build_cons(self, x, t, aff_lyap, comp, beta, delta=0):
        mv, sv = self.gp.mean_var(x[np.newaxis,:]), self.gp.sigma_var() #m+1, (m+1,m+1)
        input_dep = -(aff_lyap.act(x,t) + mv[1:]) #m
        input_indep = delta -(aff_lyap.drift(x,t) + mv[0] + comp * aff_lyap.eval(x,t)) #()
        return cp.SOC(input_dep @ self.u.T + input_indep, beta * sv[1:].T @ self.u + beta * sv[0].T)

    def eval(self, x ,t):    
        static_cost = cp.sum(self.static_costs)
        dynamic_cost = cp.sum([cost(x, t) for cost in self.dynamic_costs])
        obj = cp.Minimize(static_cost + dynamic_cost)
        cons = [constraint(x, t) for constraint in self.constraints]
        prob = cp.Problem(obj, cons)
        prob.solve(solver = 'MOSEK', warm_start=True)
        return self.u.value, [variable.value for variable in self.variables]
    
    def process(self, u):
        u, _ = u
        if u is None:
            u = np.zeros(self.m)
        return u