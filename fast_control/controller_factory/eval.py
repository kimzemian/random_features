"""Evaluation methods."""
import numpy as np


def simulate_sys(self, controller, x_0=None):
    """Simulate self.system with specified controller."""
    if x_0 is None:
        x_0 = self.x_0
    ts = np.linspace(0, self.T, self.num_steps)
    xs, us = self.system.simulate(x_0, controller, ts)
    return xs, us, ts


def eval_cs(self, controller):
    """Return true C and C_dot for simulated data with specified controller.

    can be configured to to return estimated C/C_dot
    """
    xs, us, ts = simulate_sys(self, controller)
    cs = [self.system.lyap.eval(xs[i], ts[i]) for i in range(self.num_steps)]
    c_dots = [
        self.system.lyap.eval_dot(xs[i], us[i], ts[i])
        for i in range(self.num_steps - 1)  # FIXME
    ]
    c_dots = np.concatenate(([0], c_dots))
    return np.array([cs, c_dots]), ts


def eval_all(self, controller):
    """Return true C and C_dot for simulated data with specified controller.

    can be configured to to return estimated C/C_dot
    """
    xs, us, ts = simulate_sys(self, controller)
    cs = [self.system.lyap.eval(xs[i], ts[i]) for i in range(self.num_steps)]
    c_dots = [
        self.system.lyap.eval_dot(xs[i], us[i], ts[i])
        for i in range(self.num_steps - 1)  # FIXME
    ]
    c_dots = np.concatenate(([0], c_dots))
    return np.array([cs, c_dots]), us, ts
