import sys
import os
import cvxpy as cp
import numpy as np

module_path = os.path.abspath(os.path.join('.'))
os.environ['MOSEKLM_LICENSE_FILE'] = module_path
import mosek
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path + "/core")
from core.controllers import QPController

class GPController(QPController):
    '''controller to ensure safety and/or stability using gp methods'''

    def __init__(self, system_est, gp):
        self.__name__ = gp.__name__+'controller'
        gp.train()
        self.gp = gp
        QPController.__init__(self, system_est, system_est.m)  
    
    def add_stability_constraint(self,aff_lyap, comp=0, slacked=False, time_slack = False, beta=1, coeff=0):
        if slacked:
            delta = cp.Variable()
            self.variables.append(delta)
            if time_slack:  
                self.static_costs_lambda.append(lambda t : (t+1) * coeff * cp.square(delta))
            else:
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
        static_cost = cp.sum([s_cost(t) for s_cost in self.static_costs_lambda]) + cp.sum(self.static_costs)
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
