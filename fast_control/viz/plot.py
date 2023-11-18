"""Plot methods."""

import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fast_control.controller_factory import eval_all, simulate_sys


def plot_episodic_cum_diff_from_oracle(self, data, path_to_save, c_cdot=0):
    """Plot sum of difference from the oracle controller for each controller over epochs."""
    value, val = self.set_value(c_cdot)
    fig, ax = plot_info_helper(self, *data, c_cdot)
    path = f"{path_to_save}{val}cumulative_diff_from_oracle.png"
    title = (
        "sum of difference from the oracle controller for each controller over epochs"
    )
    self.finalize_and_save(fig, ax, value, path, title)


def plot_info(self, data, cut_off, path_to_save, c_cdot=0):
    """Plot information.

    true C_dot/C for each controller over time.
    """
    value, val = self.set_value(c_cdot)
    fig, ax = plot_info_helper(self, *data, c_cdot)
    path = f"{path_to_save}true_{val}_{int(time.time())}.png"
    self.finalize_and_save(fig, ax, value, path)

    fig, ax = plot_info_helper(self, *data, c_cdot, cut_off)
    path = f"{path_to_save}true_{val}_last_{-cut_off}_sec_{int(time.time())}.png"
    self.finalize_and_save(fig, ax, value, path)


def plot_info_helper(self, gp_zs, qp_zs, model_zs, ts, c_cdot=0, cut_off=None):
    # c = 1
    if cut_off is None:
        cut_off = len(ts)
    fig, ax = plt.subplots(sharex=True)
    plt.xlabel("time (s)", fontsize=8)

    for i, gp_name in enumerate(self.gps_names):
        sns.lineplot(
            x=ts[:cut_off],
            y=gp_zs[c_cdot, i, :cut_off],
            label=gp_name + "_controller",
        )  # markersize=c

    sns.lineplot(
        x=ts[:cut_off],
        y=qp_zs[c_cdot, :cut_off],
        linestyle="dotted",
        color="black",
        label="qp_controller",
    )
    sns.lineplot(
        x=ts[:cut_off],
        y=model_zs[c_cdot, :cut_off],
        linestyle="dashdot",
        color="black",
        label="oracle_controller",
    )

    ax.legend()
    return fig, ax


def plot_simulation(self, factory, controller, x_0=None):
    """Plot simulated system trajectories using specified controller.

    one dimensional: such as inverted pendulum
    """
    xs, us, ts = factory.simulate(controller, x_0)
    ax = plt.figure(figsize=(8, 6), tight_layout=True).add_subplot(1, 1, 1)
    ax.set_xlabel("$t$", fontsize=16)
    ax.plot(ts, xs[:, 0], "-", label="theta")
    ax.plot(ts, xs[:, 1], "-", label="theta dot")
    ax.plot(ts[1:], us, "-", label="input")

    ax.legend(fontsize=16)
    ax.set_title(controller.name)
    ax.grid()
    # ax.savefig(controller_name)
    plt.show()
    plt.close()

    ax = plt.figure(figsize=(8, 6), tight_layout=True).add_subplot(1, 1, 1)
    ax.set_xlabel("$\\theta$", fontsize=16)
    ax.set_ylabel("$\dot \\theta$", fontsize=16)
    ax.plot(xs[:, 0], xs[:, 1], "-")
    ax.grid()
    plt.show()
    # ax.savefig('theta plot for'+controller_name)
    plt.close()


def plot_pred_errorbar(self, xs, ys, zs, gps):
    """Plot gp prediction on training data."""
    c = 1
    fmts = ["r.", "b.", "g.", "c."]

    plt.figure(figsize=(9, 9), dpi=240)
    for gp, fmt in zip(gps, fmts):
        plt.errorbar(
            zs,
            gp.test(xs, ys),
            gp.sigma(xs, ys),
            fmt=fmt,
            label=gp.name + " pred",
            alpha=0.4,
            markersize=c,
            capsize=3,
        )
    plt.plot(zs, zs, "k.", label="actual data", markersize=c + 1)

    plt.xlabel("$\dot{C}$")
    plt.ylabel("$\hat{\dot{C}}$")

    plt.legend()
    plt.title(f"gp predictions for {len(xs)} data points")
    plt.show()
    # plt.savefig(f'gp predictions for {len(xs)} data points')
    plt.close()


def plot_closed_loop_errorbar(self, factory, controller, gp, x_0, cut_off=0):
    """Plot error bars for simulated system given a controller."""
    c = 1
    all_xs, all_ys, all_zs = factory.training_data_gen(controller, x_0)
    xs = all_xs[cut_off:]
    ys = all_ys[cut_off:]
    zs = all_zs[cut_off:]

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        zs,
        gp.test(xs, ys),
        gp.sigma(xs, ys),
        fmt="r.",
        label=gp.name + " controller pred",
        alpha=0.4,
        markersize=c,
        capsize=3,
    )
    plt.plot(zs, zs, "k.", label="actual data", markersize=c)

    plt.xlabel("$\dot{C}$")
    plt.ylabel("$\hat{\dot{C}}$")
    plt.legend()
    plt.show()
    # plt.savefig(gp.name+' controller pred')
    plt.close()


def plot_all_closed_loop_errorbars(self, factory, controllers, gps, x_0, cut_off=0):
    """Plot error bars for simulated system for all controllers."""
    c = 1
    fmts = ["c.", "m.", "y.", "r."]
    zs_fmts = ["b.", "g.", "r.", "k."]

    plt.figure(figsize=(8, 6))
    for controller, gp, fmt, z_fmt in zip(controllers, gps, fmts, zs_fmts):
        all_xs, all_ys, all_zs = factory.training_data_gen(controller, x_0)
        xs = all_xs[cut_off:]
        ys = all_ys[cut_off:]
        zs = all_zs[cut_off:]
        plt.errorbar(
            zs,
            gp.test(xs, ys),
            gp.sigma(xs, ys),
            fmt=fmt,
            label=gp.name + " controller pred",
            alpha=0.4,
            markersize=c,
            capsize=3,
        )
        plt.plot(zs, zs, z_fmt, label="actual data", markersize=c)

    plt.xlabel("$\dot{C}$")
    plt.ylabel("$\hat{\dot{C}}$")
    plt.legend()
    plt.draw()
    # plt.savefig('all_closed_loop_errorbars')
    plt.close()


def plot_simulation_dip(self, factory, controller, x_0):
    """Plot simulated system trajectories using specified controller."""
    xs, us, ts = simulate_sys(factory, controller, x_0)
    ax = plt.figure(figsize=(8, 6), tight_layout=True).add_subplot(1, 1, 1)
    ax.set_xlabel("$t$", fontsize=16)
    ax.plot(ts, xs[:, 0], "-", label="$\\theta_1$")
    ax.plot(ts, xs[:, 1], "-", label="$\\theta_2$")
    ax.plot(ts, xs[:, 2], "-", label="$\dot \\theta_1$")
    ax.plot(ts, xs[:, 3], "-", label="$\dot \\theta_2$")
    ax.plot(ts[1:], us, "-", label="input")

    ax.legend(fontsize=16)
    ax.set_title(controller.name)
    ax.grid()
    plt.show()
    # ax.savefig(controller_name)
    plt.close()

    ax = plt.figure(figsize=(8, 6), tight_layout=True).add_subplot(1, 1, 1)
    ax.set_xlabel("$\\theta_1$", fontsize=16)
    ax.set_ylabel("$\dot \\theta_1$", fontsize=16)
    ax.plot(xs[:, 0], xs[:, 2], "-")
    ax.grid()
    # ax.savefig(controller_name+'theta 1')
    plt.show()
    plt.close()

    ax = plt.figure(figsize=(8, 6), tight_layout=True).add_subplot(1, 1, 1)
    ax.set_xlabel("$\\theta_2$", fontsize=16)
    ax.set_ylabel("$\dot \\theta_2$", fontsize=16)
    ax.plot(xs[:, 1], xs[:, 3], "-")
    ax.grid()
    # ax.savefig(controller_name+'theta 2')
    plt.show()
    plt.close()


def plot_qp(self, factory, c_cdot=0):
    """Plot QP and oracle controller."""
    qp_zs, us, ts = eval_all(factory, factory.system.qp_controller)
    model_zs, us, _ = eval_all(factory, factory.system.oracle_controller)
    _, ax = plt.subplots()
    time = 0
    ax.plot(ts[time:], qp_zs[c_cdot, :][time:], "-", label="qp_controller")
    ax.plot(ts[time:], model_zs[c_cdot, :][time:], "k-.", label="oracle_controller")
    plt.show()
    plt.close()
    _, ax = plt.subplots()
    time = 1
    ax.plot(ts[1:], us, "-", label="qp_controller_us")
    ax.plot(ts[1:], us, "k-.", label="oracle_controller_us")


def plot_mean_prediction_error(self, epochs):
    """Plot sum absolute value of true C_dot/true C minus the predicted C/C_dot over epoch."""
    with open("data/test_previous_gp.pickle", "rb") as handle:
        data = pickle.load(handle)
    value = next(iter(data.values()))
    epochs = len(value[0])
    ts = np.linspace(0, epochs, epochs)
    c = 1
    fig, ax = plt.subplots(sharex=True)
    # ax.set_ylim(0, 10000)
    ax.set_xlabel("$episodes$", fontsize=8)
    ax.set_ylabel("mean prediction error", fontsize=8)
    for key in data:
        mean_error = np.sum(data[key], axis=0)
        ax.plot(ts, mean_error, "-", label=key, markersize=c, alpha=0.4)

    ax.legend()
    ax.set_title(f"mean prediction error")
    # plt.figtext(0.12, 0.94, f"x_0={x_0},{key}")
    fig.figsize = (9, 6)
    fig.tight_layout()
    fig.savefig(f"mean prediction error")
    plt.show()
    plt.close()
