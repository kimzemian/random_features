import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import time
from fast_control.controller_factory import (
    ControllerFactory,
    create_grid_data,
    info_data_gen,
    train_controllers,
    training_data_gen,
    xdot_training_data_gen,
)
from fast_control.gp_factory import GPFactory, init_gp_dict, train_gps
from fast_control.util import load_model, save_model
from fast_control.viz import ControlSystemPlotter, plot_info
from fast_control.viz.episodic_plot import *
from fast_control.viz.plot import *


def create_grid(i):
    fixed_path = "/share/dean/fast_control/models/"
    try:
        model_path = fixed_path + f"grid_{i}/"
        control = ControllerFactory(model_path + "config.toml")
    except FileNotFoundError:
        model_path = os.makedirs(fixed_path + f"grid_{i}/")
        control = ControllerFactory(fixed_path + "config.toml")

    xs, ys, zs = create_grid_data(control, control.grid_T, control.grid_num_steps)
    np.savez(model_path + "grid.npz", xs=xs, ys=ys, zs=zs)
    controllers, gps = train_and_save_controllers(xs, ys, zs, control, model_path)

    xs, ys, zs = create_grid_data(
        control, control.grid_T, control.grid_num_steps, data_gen=xdot_training_data_gen
    )
    np.savez(model_path + "x_dot_grid.npz", xs=xs, ys=ys, zs=zs)
    x_dot_gps = train_and_save_gps(xs, ys, zs, model_path)
    return control, controllers, gps, x_dot_gps, model_path


def train_and_save_controllers(xs, ys, zs, control, path):
    gp_factory = GPFactory()
    data = init_gp_dict(gp_factory, xs, ys, zs)
    controllers, gps = train_controllers(control, gp_factory, data)
    save_model(gps, path + "gps.pickle")
    save_model(controllers, path + "controllers.pickle")
    save_model(control, path + "control.pickle")
    return controllers, gps


def train_and_save_gps(xs, ys, zs, path):
    gp_factory = GPFactory()
    data = init_gp_dict(gp_factory, xs, ys, zs)
    gps = train_gps(gp_factory, data)
    save_model(gps, path + "x_dot_gps.pickle")
    return gps


def plot_c_data(control, controllers, x_0, T, num_steps, path_to_save):
    cut_off = -num_steps // 3
    evaluations = info_data_gen(control, controllers, x_0, T, num_steps)
    plotter = ControlSystemPlotter(x_0, control.gps_names)
    plot_info(plotter, evaluations, cut_off, path_to_save, c_cdot=0)
    plot_info(plotter, evaluations, cut_off, path_to_save, c_cdot=1)


def plot_xdot_data(control, gps, x_0, T, num_steps, path_to_save):
    xs, ys, x_dots = xdot_training_data_gen(
        control, control.system.qp_controller, x_0, T, num_steps
    )
    x_dot_preds = np.empty((len(gps), len(x_dots), 4))
    for i, gp in enumerate(gps):
        x_dot_preds[i, :, :] = gp.test(xs, ys)
        # rmse = mean_squared_error(x_dots, x_dot_preds[i,:,:], squared=False, multioutput='raw_values')
        # mae = mean_absolute_error(x_dots, x_dot_preds[i,:,:], multioutput='raw_values')
        ts = np.linspace(0, T, num_steps)
    plot_xdot_helper(x_dot_preds, x_dots, ts[:-1], control.gps_names, path_to_save)


def plot_xdot_helper(preds, labels, ts, gps_names, plot_path):
    for i in range(4):
        plt.figure()
        for gp_num, name in enumerate(gps_names):
            sns.lineplot(
                x=ts,
                y=preds[gp_num, :, i],
                label=name + " $\dot{x}$ prediction",
                alpha=0.5,
            )
        sns.lineplot(
            x=ts, y=labels[:, i], color="black", linestyle="--", label="true $\dot{x}$"
        )

        plt.xlabel("time")
        plt.ylabel(f"{i}-th coordinate of x_dot")
        plt.tight_layout()
        plt.legend()
        plt.savefig(
            f"{plot_path}learn_sys_x_dot_error_{i}_{int(time.time())}.png", dpi=300
        )
        plt.show()
        plt.close()


if __name__ == "__main__":
    initial_xs = (
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([1.5, 0, 1.5, 0]),
        np.array([3.1, 0, 0, 0]),
    )
    for i in tqdm(range(6, 9)):
        control, controllers, gps, x_dot_gps, model_path = create_grid(i)
        T = control.episodic_T
        num_steps = control.episodic_num_steps
        for x_0 in initial_xs:
            plot_c_data(control, controllers, x_0, T, num_steps, model_path)
            plot_xdot_data(control, x_dot_gps, x_0, T, num_steps, model_path)
