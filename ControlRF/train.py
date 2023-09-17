import pickle
import numpy as np
import torch
from tqdm import tqdm
from ControlRF import (
    GPController,
    ADPKernel,
    ADPRandomFeatures,
    ADKernel,
    ADRandomFeatures,
)
from ControlRF.data import info_data_gen, training_data_gen


def init_kernel(gp_name, datum, sgm, m, rf_d):
    """initialize specified kernel"""
    # match gp_name:
    #     case ADRandomFeatures.__name__:
    #         return ADRandomFeatures(*datum, sgm, m*rf_d)
    #     case ADPRandomFeatures.__name__:
    #         return ADPRandomFeatures(*datum, sgm, rf_d)
    #     case ADKernel.__name__:
    #         return ADKernel(*datum, sgm)
    #     case ADPKernel.__name__:
    #         return ADPKernel(*datum, sgm)

    if gp_name == ADRandomFeatures.name:
        return ADRandomFeatures(*datum, sgm, m * rf_d)
    elif gp_name == ADPRandomFeatures.name:
        return ADPRandomFeatures(*datum, sgm, rf_d)
    elif gp_name == ADKernel.name:
        return ADKernel(*datum, sgm)
    elif gp_name == ADPKernel.name:
        return ADPKernel(*datum, sgm)


def train(system_est, data, sgm=1, slack="linear", D=2, coeff=1e6):
    """generates and trains data driven controllers given training data"""
    _, ys, zs = next(iter(data.values()))
    num = len(zs) // 10
    m = len(ys[0]) - 1
    rf_d = num + 1 if num % 2 else num  # TODO:make rf_d user determinable

    gps = []
    controllers = []
    for gp_name, datum in data.items():
        gps.append(init_kernel(gp_name, datum, sgm, m, rf_d))

    for gp in gps:
        print(gp.name)
        gp_controller = GPController(system_est, gp)
        if D == 2:
            gp_controller.add_regularizer(system_est.fb_lin, 25)
            gp_controller.add_static_cost(np.identity(2))
        else:
            gp_controller.add_static_cost(np.identity(1))

        gp_controller.add_stability_constraint(
            system_est.lyap, comp=system_est.alpha, slack=slack, coeff=coeff
        )
        controllers.append(gp_controller)
        print(f"training time for {gp.name}_gp is: {gp.training_time}")
    return controllers, gps


def train_episodic(
    system,
    system_est,
    x_0,
    epochs,
    T,
    num_steps,
    gps_names,
    info=False,
    func=None,
    sgm=1,
    slack="linear",
    D=2,
    coeff=1e6,
):
    """episodically trains data driven controllers given initial training data, for specified epochs, info saves more information"""
    xs, ys, zs = training_data_gen(
        system, system_est, system.qp_controller, torch.from_numpy(x_0), T, num_steps
    )
    data = dict.fromkeys(gps_names)
    tested_data = dict.fromkeys(gps_names)
    for gp in gps_names:
        data[gp] = (np.copy(xs), np.copy(ys), np.copy(zs))
        tested_data[gp] = np.empty((len(zs), epochs))

    controllers, gps = train(system_est, data, sgm, slack, D, coeff)

    if info:
        gp_zs = np.empty((len(gps_names), num_steps, epochs))
        qp_zs = np.empty((num_steps, epochs))
        model_zs = np.empty((num_steps, epochs))
        diff_gp_zs = np.empty((len(gps_names), epochs))
        diff_qp_zs = np.empty(epochs)

    for epoch in tqdm(range(epochs)):
        for controller, gp in zip(controllers, gps):
            # print(controller.name,gp.name)
            x, y, z = training_data_gen(
                system, system_est, controller, xs[0], T, num_steps
            )
            # print(z)
            xs, ys, zs = data[gp.name]
            xs = np.concatenate((xs, x))
            ys = np.concatenate((ys, y))
            zs = np.concatenate((zs, z))
            data[gp.name] = xs, ys, zs

            pred = gp.test(x, y)
            tested_data[gp.name][:, epoch] = np.abs(pred - z)

        controllers, gps = train(system_est, data, sgm, slack, D)
        print(f"iteration{epoch}:data size:{len(xs)}")

        # plot information for all epochs
        if info:
            # compute func(true c_dot/true c) for controllers
            gp_z, qp_z, model_z, ts = info_data_gen(
                xs[0], controllers, system, T, num_steps, func
            )
            gp_zs[:, :, epoch] = gp_z
            qp_zs[:, epoch] = qp_z
            model_zs[:, epoch] = model_z

            # computes sum of difference between the controllers and the oracle
            diff_gp_z, diff_qp_z = info_data_gen(
                xs[0], controllers, system, T, num_steps, func, info
            )
            diff_gp_zs[:, epoch] = diff_gp_z
            diff_qp_zs[epoch] = diff_qp_z

    # saves sum of difference between the controllers and the oracle
    if info:
        np.savez(
            f"data/{func.__name__}", gp_zs=gp_zs, qp_zs=qp_zs, model_zs=model_zs, ts=ts
        )
        np.savez(
            "data/diff_from_oracle",
            gp_zs=diff_gp_zs,
            qp_zs=diff_qp_zs,
            model_zs=np.zeros(epochs),
            ts=np.arange(epochs),
        )
        with open("data/test_previous_gp.pickle", "wb") as handle:
            pickle.dump(tested_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return controllers, gps
