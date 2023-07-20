import sys
import mosek
import numpy as np

from ControlRF.train import *
from ControlRF.viz.episodic_plot import *
from ControlRF.viz.animate import *
from ControlRF.viz.plot import *
from ControlRF.util import *
from core.systems import DoubleInvertedPendulum

np.set_printoptions(threshold=sys.maxsize)


    



if __name__ == "__main__":
    
    system_est = DoubleInvertedPendulum(2, 2, .8, .8)
    system = DoubleInvertedPendulum(0.5, 0.5, 1, 2)
    init_controllers(system, system_est)
    
    x_0 = np.array([3.1, 0, 0, 0])
    T = 10
    num_steps = 100
    epochs = 20
    slacks = ['constant', 'linear', 'quadratic']
    sgms = [5,10,15,20,25,30]
    
    
    for sgm in sgms:
        for slack in slacks:
            controllers, gps = train_episodic(system, system_est, x_0, epochs, T, num_steps, info=True, func=eval_c, sgm=sgm, slack=slack)
            data = np.load('data/_diff from oracle.npz')
            print(f'sgm={sgm}, slack={slack},diff={data["gp_zs"].T[-1]}')