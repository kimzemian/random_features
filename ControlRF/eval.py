import numpy as np


def simulate(system, controller, x_0, T, num_steps):
    """simulate system with specified controller"""

    ts = np.linspace(0, T, num_steps)
    xs, us = system.simulate(x_0, controller, ts)
    return xs, us, ts


def eval_c_dot(system, controller, x_0, T, num_steps):
    """returns estimated/true C_dot for simulated data with specified controller"""
    xs, us, ts = simulate(system, controller, x_0, T, num_steps)
    zs = [system.lyap.eval_dot(xs[i], us[i], ts[i]) for i in range(len(us))]
    return np.array(zs)


def eval_c(system, controller, x_0, T, num_steps):
    """returns estimated/true C for simulated data with specified controller"""
    xs, _, ts = simulate(system, controller, x_0, T, num_steps)
    zs = [system.lyap.eval(xs[i], ts[i]) for i in range(num_steps)]
    return np.array(zs)
