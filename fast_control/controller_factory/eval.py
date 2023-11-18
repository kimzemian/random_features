"""Evaluation methods."""
import numpy as np


def simulate_sys(self, controller, x_0=None, T=None, num_steps=None):
    """Simulate self.system with specified controller."""
    ts = np.linspace(0, T, num_steps)
    xs, us = self.system.simulate(x_0, controller, ts)
    return xs, us, ts


def eval_cs(self, controller, x_0=None, T=None, num_steps=None):
    """Return true C and C_dot for simulated data with specified controller.

    can be configured to to return estimated C/C_dot
    """
    xs, us, ts = simulate_sys(self, controller, x_0, T, num_steps)
    cs = [self.system.lyap.eval(xs[i], ts[i]) for i in range(num_steps)]
    c_dots = [
        self.system.lyap.eval_dot(xs[i], us[i], ts[i])
        for i in range(num_steps - 1)  # FIXME
    ]
    c_dots = np.concatenate(([0], c_dots))
    return np.array([cs, c_dots]), ts


def eval_all(self, controller, x_0=None, T=None, num_steps=None):
    """Return true C and C_dot for simulated data with specified controller.

    can be configured to to return estimated C/C_dot
    """
    xs, us, ts = simulate_sys(self, controller, x_0, T, num_steps)
    cs = [self.system.lyap.eval(xs[i], ts[i]) for i in range(num_steps)]
    c_dots = [
        self.system.lyap.eval_dot(xs[i], us[i], ts[i])
        for i in range(num_steps - 1)  # FIXME
    ]
    c_dots = np.concatenate(([0], c_dots))
    return np.array([cs, c_dots]), us, ts
