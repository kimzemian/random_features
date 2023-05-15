import numpy as np
import numpy.linalg as la
import torch
from tqdm import tqdm
from ControlRF import (
    GPController,
    ADPKernel,
    ADPRandomFeatures,
    ADKernel,
    ADRandomFeatures,
)

def simulate(system, controller, x_0, T, num_steps):
    """simulate system with specified controller"""

    ts = np.linspace(0, T, num_steps)
    xs, us = system.simulate(x_0, controller, ts)
    return xs, us, ts


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


def training_data_gen(system, controller, lyap, lyap_est, x_0, T, num_steps):
    """generates training data given a controller"""

    xs, us, ts = simulate(system, controller, x_0, T, num_steps)
    xs, ys, zs = build_ccf_data(lyap, lyap_est, xs, us, ts)  # ts=ts[1:-1]
    return xs, ys, zs


def train(system_est, lyap_est, fb_lin, alpha, xs, ys, zs):
    '''generates and trains data driven controllers given training data'''
    num = len(zs)//10
    rf_d = num+1 if num%2 else num
    ad_rf = ADRandomFeatures(xs, ys, zs, rf_d=rf_d)
    adp_rf = ADPRandomFeatures(xs, ys, zs, rf_d=rf_d)
    ad_kernel = ADKernel(xs, ys, zs)
    adp_kernel = ADPKernel(xs, ys, zs)
    gps = [ad_kernel, adp_rf, adp_kernel, ad_rf]
    controllers = []
    gps_name = []
    for gp in gps:
        gps_name.append(gp.__name__)
        print(gp.__name__)
        gp_controller = GPController(system_est, gp)
        gp_controller.add_regularizer(fb_lin, 25)
        gp_controller.add_static_cost(np.identity(2))
        gp_controller.add_stability_constraint(
            lyap_est, comp=alpha, slacked=True, time_slack=True, coeff=1e5
        )
        controllers.append(gp_controller)
        print(f"training time for {gp.__name__}_gp is: {gp.training_time}")
    return controllers, gps, gps_name


def train_episodic(system, system_est, lyap, lyap_est, fb_lin, alpha, xs, ys, zs, epochs, T, num_steps, info=False, qp_controller=None, oracle_controller=None, func=None):
    '''episodically trains data driven controllers given initial training data, for specified epochs, info option saves more information'''
    
    controllers, gps, gps_name = train(system_est, lyap_est, fb_lin, alpha, xs, ys, zs)
    data = dict.fromkeys(gps_name,(xs, ys, zs))
    if info:
        gp_zs = np.empty((4,epochs))
        qp_zs = np.empty(epochs)
    for epoch in tqdm(range(epochs)):
        if info:
            gp_z, qp_z = info_data_gen(xs[0], controllers, system, lyap_est, T, num_steps, qp_controller, oracle_controller, lyap, eval_c, info)
            gp_zs[:,epoch] = gp_z
            qp_zs[epoch] = qp_z
        for controller,gp in zip(controllers,gps):
            x, y, z = training_data_gen(system, controller, lyap, lyap_est, xs[0], T, num_steps)
            xs, ys, zs = data[gp.__name__]
            xs = np.concatenate((xs, x))
            ys = np.concatenate((ys, y))
            zs = np.concatenate((zs, z))
            data[gp.__name__] = xs, ys, zs
        controllers, gps, _ = train(system_est, lyap_est, fb_lin, alpha, xs, ys, zs)
        print(f"iteration{epoch}:data size:{len(xs)}")
    if info:
        np.savez(f"data/diff", 
            gp_zs=gp_zs,
            qp_zs=qp_zs,
            model_zs=np.zeros(epochs),
            ts=np.arange(epochs))
    return controllers, gps

def create_grid_data(system, qp_controller, lyap_est, T, num_steps):
    initial_x0s = (
        np.mgrid[0.1 : np.pi : 1, -1:1.1:0.4, 0 : np.pi : 1, -1:1.1:0.4]
        .reshape(4, -1)
        .T
    )
    xs, ys, zs = training_data_gen(
        system, qp_controller, lyap_est, torch.from_numpy(initial_x0s[0]), T, num_steps
    )
    for x_0 in initial_x0s[1:]:
        x, y, z = training_data_gen(
            system, qp_controller, lyap_est, torch.from_numpy(x_0), T, num_steps
        )
        xs = np.concatenate((xs, x))
        ys = np.concatenate((ys, y))
        zs = np.concatenate((zs, z))

    np.savez(f"data_{T}_{num_steps},{xs.shape}", xs=xs, ys=ys, zs=zs)
    return None

def eval_c_dot(system, controller, aff_lyap, x_0, T, num_steps):
    """returns estimated/true C_dot for simulated data with specified controller"""
    xs, us, ts = simulate(system, controller, x_0, T, num_steps)
    zs = [aff_lyap.eval_dot(xs[i], us[i], ts[i]) for i in range(len(us))]
    return np.array(zs)


def eval_c(system, controller, aff_lyap, x_0, T, num_steps):
    """returns estimated/true C for simulated data with specified controller"""
    xs, _, ts = simulate(system, controller, x_0, T, num_steps)
    zs = [aff_lyap.eval(xs[i], ts[i]) for i in range(num_steps)]
    return np.array(zs)


def info_data_gen(
    x_0,
    controllers,
    system,
    lyap_est,
    T,
    num_steps,
    qp_controller,
    oracle_controller,
    lyap,
    func,
    info=False
):
    """runs all controllers to generate data"""
    ts = np.linspace(0, T, num_steps)
    gp_zs = np.empty((4, len(ts)))
    for i, controller in enumerate(controllers):
        # print('--------------------------------------------------------')
        # print(controller.__name__)
        gp_zs[i, :] = func(system, controller, lyap, x_0, T, num_steps)#is this correct?

    qp_zs = func(system, qp_controller, lyap, x_0, T, num_steps)#is this correct?

    model_zs = func(system, oracle_controller, lyap, x_0, T, num_steps)
    
    if not info:
        np.savez(
            f"data/[0,pi,0,0]/{func.__name__}",
            gp_zs=gp_zs,
            qp_zs=qp_zs,
            model_zs=model_zs,
            ts=ts
        )
    else:
        return la.norm(gp_zs-model_zs, axis=1), la.norm(qp_zs-model_zs)
    return None
