"""Trains data driven gps and controllers."""
import pickle

import numpy as np
from tqdm import tqdm

from .data import (
    create_grid_data,
    info_data_gen,
    training_data_gen,
)
from .init_controllers import init_gp_dict, init_gpcontroller_pairs


def train_grid(self):
    """Train data driven controllers on grid data."""
    xs, ys, zs = create_grid_data(self)
    data = init_gp_dict(xs, ys, zs)
    controllers, gps = init_gpcontroller_pairs(self, data)
    return controllers, gps


def train_episodic_with_info(self):
    """Episodically train data driven controllers and save information."""
    xs, ys, zs = training_data_gen(self, self.system.qp_controller)
    data = dict.fromkeys(self.gps_names)
    tested_data = dict.fromkeys(self.gps_names)
    for gp_name in self.gps_names:
        data[gp_name] = (np.copy(xs), np.copy(ys), np.copy(zs))
        tested_data[gp_name] = np.empty((self.num_steps - 1, self.epochs))

    controllers, gps = init_gpcontroller_pairs(self, data)

    gp_zs = np.empty((2, len(self.gps_names), self.num_steps, self.epochs))
    qp_zs = np.empty((2, self.num_steps, self.epochs))
    model_zs = np.empty((2, self.num_steps, self.epochs))
    diff_gp_zs = np.empty((2, len(self.gps_names), self.epochs))
    diff_qp_zs = np.empty((2, self.epochs))
    # us = np.empty((len(self.gps_names), self.num_steps - 1, self.epochs))
    for epoch in tqdm(range(self.epochs)):
        for controller, gp in zip(controllers, gps):
            # print(controller.name,gp.name)
            x, y, z = training_data_gen(self, controller)
            xs, ys, zs = data[gp.name]
            xs = np.concatenate((xs, x))
            ys = np.concatenate((ys, y))
            zs = np.concatenate((zs, z))
            data[gp.name] = xs, ys, zs
            pred = gp.test(x, y)
            tested_data[gp.name][:, epoch] = np.abs(pred - z)

        controllers, gps = init_gpcontroller_pairs(self, data)
        print(f"iteration {epoch} training ended")

        # compute eval_func(true c_dot/true c) for controllers
        gp_z, qp_z, model_z, ts = info_data_gen(self, controllers)
        gp_zs[:, :, :, epoch] = gp_z
        qp_zs[:, :, epoch] = qp_z
        model_zs[:, :, epoch] = model_z

        # computes sum of difference between the controllers and the oracle
        diff_gp_z, diff_qp_z = info_data_gen(self, controllers, info=True)
        diff_gp_zs[:, :, epoch] = diff_gp_z
        diff_qp_zs[:, epoch] = diff_qp_z

    # saves sum of difference between the controllers and the oracle
    np.savez("data/eval_cs", gp_zs=gp_zs, qp_zs=qp_zs, model_zs=model_zs, ts=ts)
    np.savez(
        "data/diff_from_oracle",
        gp_zs=diff_gp_zs,
        qp_zs=diff_qp_zs,
        model_zs=np.zeros((2, self.epochs)),
        ts=np.arange(self.epochs),
    )
    with open("data/test_previous_gp.pickle", "wb") as handle:
        pickle.dump(tested_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return controllers, gps


def train_episodic(self):
    """Episodically train data driven controllers."""
    xs, ys, zs = training_data_gen(self.system.qp_controller)
    data = init_gp_dict(self, xs, ys, zs)

    controllers, gps = init_gpcontroller_pairs(self, data)

    for epoch in tqdm(range(self.epochs)):
        for controller, gp in zip(controllers, gps):
            # print(controller.name,gp.name)
            x, y, z = training_data_gen(self, controller)
            # print(z)
            xs, ys, zs = data[gp.name]
            xs = np.concatenate((xs, x))
            ys = np.concatenate((ys, y))
            zs = np.concatenate((zs, z))
            data[gp.name] = xs, ys, zs

        controllers, gps = init_gpcontroller_pairs(self, data)
        print(f"iteration{epoch}")
    return controllers, gps
