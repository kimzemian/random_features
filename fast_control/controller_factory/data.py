"""Methods that generate/save data."""
import numpy as np
import numpy.linalg as la
import torch
from .eval import simulate_sys, eval_cs, eval_all


def build_ccf_data(self, xs, us, ts):
    """Estimate error in the derivate of the CCF function.

    uses forward differencing
    """
    av_x = (xs[:-1] + xs[1:]) / 2
    zs = [
        (
            self.system.lyap.eval(xs[i + 1], ts[i + 1])
            - self.system.lyap.eval(xs[i], ts[i])
        )
        / (ts[i + 1] - ts[i])
        - self.system_est.lyap.eval_dot(av_x[i], us[i], ts[i])
        for i in range(len(us))
    ]
    ys = np.concatenate((np.ones((len(us), 1)), us), axis=1)
    return av_x, ys, zs


def training_data_gen(self, controller, x_0=None):
    """Generate training data given a controller."""
    xs, us, ts = simulate_sys(self, controller, x_0)
    xs, ys, zs = build_ccf_data(self, xs, us, ts)  # ts=ts[1:-1]
    return xs, ys, zs


def create_grid_data(self):
    """Initialize grided training data.

    run a grid of initial points with nominal controller for several steps
    """
    initial_x0s = (
        np.mgrid[0.1 : np.pi : 1, -1:1.1:0.4, 0 : np.pi : 1, -1:1.1:0.4]
        .reshape(4, -1)
        .T
    )
    for i, x_0 in enumerate(initial_x0s):
        if i == 0:
            xs, ys, zs = training_data_gen(
                self,
                self.system.qp_controller,
                torch.from_numpy(x_0),
            )
        else:
            x, y, z = training_data_gen(
                self,
                self.system.qp_controller,
                torch.from_numpy(x_0),
            )
            xs = np.concatenate((xs, x))
            ys = np.concatenate((ys, y))
            zs = np.concatenate((zs, z))

    x, y, z = training_data_gen(
        self, self.system.qp_controller, torch.FloatTensor([0.1, 0, 0, 0])
    )
    xs = np.concatenate((xs, x))
    ys = np.concatenate((ys, y))
    zs = np.concatenate((zs, z))

    np.savez("data/init_grid", xs=xs, ys=ys, zs=zs)
    return xs, ys, zs


def info_data_gen(self, controllers, info=False):
    """Run controllers to generate data.

    eval_func options: eval_c,eval_c_dot
    info returns norm of (true c_dot for gp/qp controller - oracle controller)
    """
    gp_cs = np.empty((2, len(controllers), self.num_steps))
    for i, controller in enumerate(controllers):
        gp_cs[:, i, :], ts = eval_cs(self, controller)

    qp_cs, _ = eval_cs(self, self.system.qp_controller)
    model_cs, _ = eval_cs(self, self.system.oracle_controller)
    if info:
        return la.norm(gp_cs - model_cs[:, np.newaxis, :], axis=2), la.norm(
            qp_cs - model_cs, axis=1
        )

    return gp_cs, qp_cs, model_cs, ts


def info_data_gen_with_u(self, controllers, info=False):
    """Run controllers to generate data.

    eval_func options: eval_c,eval_c_dot
    info returns norm of (true c_dot for gp/qp controller - oracle controller)
    """
    gp_cs = np.empty((2, len(controllers), self.num_steps))
    for i, controller in enumerate(controllers):
        gp_cs[:, i, :], us, ts = eval_all(self, controller)

    qp_cs, _, _ = eval_all(self, self.system.qp_controller)
    model_cs, _, _ = eval_all(self, self.system.oracle_controller)
    if info:
        return la.norm(gp_cs - model_cs[:, np.newaxis, :], axis=2), la.norm(
            qp_cs - model_cs, axis=1
        )

    return gp_cs, qp_cs, model_cs, us, ts


def grid_info(self, controllers):
    """Wrap info_data_gen."""
    # compute eval_func(true c_dot/true c) for controllers
    gp_z, qp_z, model_z, ts = info_data_gen(self, controllers)
    np.savez("data/eval_cs", gp_z=gp_z, qp_z=qp_z, model_z=model_z, ts=ts)
