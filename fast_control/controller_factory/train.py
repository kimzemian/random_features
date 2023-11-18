"""Trains data driven gps and controllers."""
import pickle

import numpy as np
from tqdm import tqdm

from .data import (
    create_grid_data,
    info_data_gen,
    training_data_gen,
)
from .init_controllers import train_controllers
from fast_control.gp_factory import init_gp_dict, GPFactory


# def train_grid(self): #FIXME update create_grid_data
#     """Train data driven controllers on grid data."""
#     xs, ys, zs = create_grid_data(self)
#     gp_factory = GPFactory()
#     data = init_gp_dict(gp_factory, xs, ys, zs)
#     controllers, gps = train_controllers(self, gp_factory, data)
#     return controllers, gps


def train_episodic_with_info(self, data_path, warm_start=False):
    """Episodically train data driven controllers and save information."""
    if warm_start:
        init_data = np.load("/home/kk983/fast_control/data/grid/init_grid_small.npz")
        xs, ys, zs = init_data['xs'], init_data['ys'], init_data['zs']
    else:
        xs, ys, zs = training_data_gen(self, self.system.qp_controller)
    gp_factory = GPFactory()
    self.gps_names = gp_factory.gp_param_dict.keys()
    data = dict.fromkeys(self.gps_names)
    tested_data = dict.fromkeys(self.gps_names)
    for gp_name in self.gps_names:
        data[gp_name] = (np.copy(xs), np.copy(ys), np.copy(zs))
        tested_data[gp_name] = np.empty((self.num_steps - 1, self.epochs))

    controllers, gps = train_controllers(self, gp_factory, data)

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

        controllers, gps = train_controllers(self, gp_factory, data)
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
    with open(data_path+"raw_episodic_data.pkl", "wb") as handle:
        pickle.dump(data, handle)
    np.savez(data_path+"eval_cs", gp_zs=gp_zs, qp_zs=qp_zs, model_zs=model_zs, ts=ts)
    np.savez(
        data_path+"diff_from_oracle",
        gp_zs=diff_gp_zs,
        qp_zs=diff_qp_zs,
        model_zs=np.zeros((2, self.epochs)),
        ts=np.arange(self.epochs),
    )
    with open(data_path+"test_previous_gp.pickle", "wb") as handle:
        pickle.dump(tested_data, handle)

    return controllers, gps


def train_episodic(self):
    """Episodically train data driven controllers."""
    xs, ys, zs = training_data_gen(self.system.qp_controller)
    gp_factory = GPFactory()
    data = init_gp_dict(gp_factory, xs, ys, zs)

    controllers, gps = train_controllers(self, gp_factory, data)

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

        controllers, gps = train_controllers(self, gp_factory, data)
        print(f"iteration{epoch}")
    return controllers, gps
