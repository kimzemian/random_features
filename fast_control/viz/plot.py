"""Plot methods."""
import re
import time

import matplotlib.pyplot as plt
import numpy as np
from fast_control.controller_factory import simulate_sys, eval_cs, eval_all


def plot_info(x_0, controllers, path, diff=False, c_cdot=0):
    """Plot information.

    options:
    1.true C_dot/C for each controller over time
    2.sum of difference from the oracle controller for each controller over epochs
    """
    c = 1
    fmts = ["c-", "m-", "y-", "r-"]

    fig, ax = plt.subplots(sharex=True)
    ax.set_xlabel("$t$", fontsize=8)
    ax.set_ylabel("$C$", fontsize=8)
    data = np.load(path)

    if diff:
        gp_zs = data["gp_zs"]
        qp_zs = data["qp_zs"]
        model_zs = data["model_zs"]
        ts = data["ts"]
    else:
        gp_zs = data["gp_zs"][:, :, :, -1]
        qp_zs = data["qp_zs"][:, :, -1]
        model_zs = data["model_zs"][:, :, -1]
        ts = data["ts"]

    for i, (controller, fmt) in enumerate(zip(controllers, fmts)):
        ax.plot(ts, gp_zs[c_cdot, i], fmt, label=controller.name, markersize=c)

    ax.plot(ts, qp_zs[c_cdot, :], "-", label="qp_controller", markersize=c)
    ax.plot(ts, model_zs[c_cdot, :], "k-.", label="oracle_controller", markersize=c)

    ax.legend()
    if diff:
        ax.set_title("difference from oracle C/C_dot for controllers over episodes")
    else:
        ax.set_title("True C/C_dot for controllers over time")
    plt.figtext(0.12, 0.94, f"x_0={x_0}")
    fig.figsize = (9, 6)
    fig.tight_layout()
    fig.savefig(f"plots/acrobat/_10_sec-time{int(time.time())}.png")
    plt.show()
    plt.close()

    if not diff:
        fig, ax = plt.subplots(sharex=True)
        ax.set_xlabel("$t$", fontsize=8)
        ylabel = "$C$" if c_cdot == 0 else "$C^{dot}$"
        ax.set_ylabel(ylabel, fontsize=8)

        t_1 = 170
        t_2 = 200
        for i, (controller, fmt) in enumerate(zip(controllers, fmts)):
            ax.plot(
                ts[t_1:t_2],
                gp_zs[c_cdot, i, t_1:t_2],
                fmt,
                label=controller.name,
                markersize=c,
                alpha=0.4,
            )
        ax.plot(
            ts[t_1:t_2],
            qp_zs[c_cdot, t_1:t_2],
            "-",
            label="qp_controller",
            markersize=c,
        )
        ax.plot(
            ts[t_1:t_2],
            model_zs[c_cdot, t_1:t_2],
            "k-.",
            label="oracle_controller",
            markersize=c,
        )

        ax.legend()
        ax.set_title(f"True C/C_dot for controllers over time {t_1}:{t_2}")
        plt.figtext(0.12, 0.94, f"x_0={x_0}")
        fig.figsize = (9, 6)
        fig.tight_layout()
        fig.savefig(f"plots/acrobat/sec_{t_1}_{t_2}-time{int(time.time())}.png")
        plt.show()

        plt.close()


def plot_simulation(factory, controller, x_0=None):
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


def plot_pred_errorbar(xs, ys, zs, gps):
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


def plot_closed_loop_errorbar(factory, controller, gp, x_0, cut_off=0):
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


def plot_all_closed_loop_errorbars(factory, controllers, gps, x_0, cut_off=0):
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


def plot_simulation_dip(factory, controller, x_0):
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


def plot_qp(factory, c_cdot=0):
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
