import numpy as np
import numpy.linalg as la
import torch

from ControlRF.eval import simulate


def build_ccf_data(lyap, lyap_est, xs, us, ts):
    """estimate error in the derivate of the CCF function
    using forward differencing"""

    av_x = (xs[:-1] + xs[1:]) / 2
    zs = [
        (lyap.eval(xs[i + 1], ts[i + 1]) - lyap.eval(xs[i], ts[i]))
        / (ts[i + 1] - ts[i])
        - lyap_est.eval_dot(av_x[i], us[i], ts[i])
        for i in range(len(us))
    ]
    ys = np.concatenate((np.ones((len(us), 1)), us), axis=1)
    return av_x, ys, zs


def training_data_gen(system, system_est, controller, x_0, T, num_steps):
    """generates training data given a controller"""
    xs, us, ts = simulate(system, controller, x_0, T, num_steps)
    xs, ys, zs = build_ccf_data(system.lyap, system_est.lyap, xs, us, ts)  # ts=ts[1:-1]
    return xs, ys, zs


def create_grid_data(system, system_est, T, num_steps):
    initial_x0s = (
        np.mgrid[0.1 : np.pi : 1, -1:1.1:0.4, 0 : np.pi : 1, -1:1.1:0.4]
        .reshape(4, -1)
        .T
    )
    xs, ys, zs = training_data_gen(
        system,
        system_est,
        system.qp_controller,
        torch.from_numpy(initial_x0s[0]),
        T,
        num_steps,
    )
    for x_0 in initial_x0s[1:]:
        x, y, z = training_data_gen(
            system,
            system_est,
            system.qp_controller,
            torch.from_numpy(x_0),
            T,
            num_steps,
        )
        xs = np.concatenate((xs, x))
        ys = np.concatenate((ys, y))
        zs = np.concatenate((zs, z))

    np.savez(f"data/grid_{T}_{num_steps},{xs.shape}", xs=xs, ys=ys, zs=zs)
    return None


def info_data_gen(x_0, controllers, system, T, num_steps, func, info=False):
    """runs all controllers to generate data
    info returns norm of the difference in gp/qp controller
    true C/C_dot and oracle controller
    """
    ts = np.linspace(0, T, num_steps)
    gp_zs = np.empty((len(controllers), num_steps))
    for i, controller in enumerate(controllers):
        gp_zs[i, :] = func(system, controller, x_0, T, num_steps)

    qp_zs = func(system, system.qp_controller, x_0, T, num_steps)

    model_zs = func(system, system.oracle_controller, x_0, T, num_steps)

    if info:
        return la.norm(gp_zs - model_zs, axis=1), la.norm(qp_zs - model_zs)

    return gp_zs, qp_zs, model_zs, ts
