import re
import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


plt.style.use("seaborn-whitegrid")

from ControlRF.util import *


def plot_info(x_0, controllers, path, diff=False):
    """plotting true C_dot/C using controllers over time or
    plots sum of difference from the oracle controller for each controller over epochs"""

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
        gp_zs = data["gp_zs"][:,:,-1]
        qp_zs = data["qp_zs"][:,-1]
        model_zs = data["model_zs"][:,-1]
        ts = data["ts"]

    for i, (controller, fmt) in enumerate(zip(controllers, fmts)):
        ax.plot(ts, gp_zs[i], fmt, label=controller.__name__, markersize=c)

    ax.plot(ts, qp_zs, "-", label="qp_controller", markersize=c)
    ax.plot(ts, model_zs, "k-.", label="oracle_controller", markersize=c)

    ax.legend()
    if diff:
        ax.set_title(f'difference from oracle C/C_dot for controllers over episodes')
    else: 
        ax.set_title(f'True C/C_dot for controllers over time')
    plt.figtext(0.12, 0.94, f"x_0={x_0}")
    fig.figsize = (9, 6)
    fig.tight_layout()
    fig.savefig(f"dip_plots/{re.sub('.',',',str(x_0))}_10_sec.png")
    plt.show()
    plt.close()
    
    if not diff:
        
        fig, ax = plt.subplots(sharex=True)
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel("$C$", fontsize=8)

        t_1 = 80
        t_2 = 100
        for i, (controller, fmt) in enumerate(zip(controllers, fmts)):
            ax.plot(
                ts[t_1:t_2],
                gp_zs[i, t_1:t_2],
                fmt,
                label=controller.__name__,
                markersize=c,
                alpha=0.4,
            )
        ax.plot(ts[t_1:t_2], qp_zs[t_1:t_2], "-", label="qp_controller", markersize=c)
        ax.plot(ts[t_1:t_2], model_zs[t_1:t_2], "k-.", label="oracle_controller", markersize=c)

        ax.legend()
        ax.set_title(f'True C/C_dot for controllers over time {t_1}:{t_2}')
        plt.figtext(0.12, 0.94, f"x_0={x_0}")
        fig.figsize = (9, 6)
        fig.tight_layout()
        fig.savefig(f"dip_plots/sec_{t_1}_{t_2}.png")
        plt.show()

        plt.close()
    

    


def plot_simulation(system, controller, controller_name, x_0, T=20, num_steps=200):
    xs, us, ts = simulate(system, controller, x_0, T, num_steps)
    """plotting simulated system trajectories using specified controller
    one dimensional: such as inverted pendulum"""

    ax = plt.figure(figsize=(8, 6), tight_layout=True).add_subplot(1, 1, 1)
    ax.set_xlabel("$t$", fontsize=16)
    ax.plot(ts, xs[:, 0], "-", label="theta")
    ax.plot(ts, xs[:, 1], "-", label="theta dot")
    ax.plot(ts[1:], us, "-", label="input")

    ax.legend(fontsize=16)
    ax.set_title(controller_name)
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
    """plotting gp prediction on training data"""

    c = 1
    fmts = ["r.", "b.", "g.", "c."]

    plt.figure(figsize=(9, 9), dpi=240)
    for gp, fmt in zip(gps, fmts):
        plt.errorbar(
            zs,
            gp.test(xs, ys),
            gp.sigma(xs, ys),
            fmt=fmt,
            label=gp.__name__ + " pred",
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


def plot_closed_loop_errorbar(
    system, aff_lyap, controller, gp, x_0, cut_off=0, T=20, num_steps=200
):
    """plotting error bars for simulated system with specified controller"""
    c = 1
    all_xs, all_ys, all_zs = data_gen(system, controller, aff_lyap, x_0, T, num_steps)
    xs = all_xs[cut_off:]
    ys = all_ys[cut_off:]
    zs = all_zs[cut_off:]

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        zs,
        gp.test(xs, ys),
        gp.sigma(xs, ys),
        fmt="r.",
        label=gp.__name__ + " controller pred",
        alpha=0.4,
        markersize=c,
        capsize=3,
    )
    plt.plot(zs, zs, "k.", label="actual data", markersize=c)

    plt.xlabel("$\dot{C}$")
    plt.ylabel("$\hat{\dot{C}}$")
    plt.legend()
    plt.show()
    # plt.savefig(gp.__name__+' controller pred')
    plt.close()


def plot_all_closed_loop_errorbars(
    system, aff_lyap, controllers, gps, x_0, cut_off=0, T=20, num_steps=200
):
    """plotting error bars for simulated system for all controllers"""
    c = 1
    fmts = ["c.", "m.", "y.", "r."]
    zs_fmts = ["b.", "g.", "r.", "k."]

    plt.figure(figsize=(8, 6))
    for controller, gp, fmt, z_fmt in zip(controllers, gps, fmts, zs_fmts):
        all_xs, all_ys, all_zs = data_gen(
            system, controller, aff_lyap, x_0, T, num_steps
        )
        xs = all_xs[cut_off:]
        ys = all_ys[cut_off:]
        zs = all_zs[cut_off:]
        plt.errorbar(
            zs,
            gp.test(xs, ys),
            gp.sigma(xs, ys),
            fmt=fmt,
            label=gp.__name__ + " controller pred",
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


def plot_simulation_dip(system, controller, controller_name, x_0, T=20, num_steps=200):
    xs, us, ts = simulate(system, controller, x_0, T, num_steps)
    """plotting simulated system trajectories using specified controller """

    ax = plt.figure(figsize=(8, 6), tight_layout=True).add_subplot(1, 1, 1)
    ax.set_xlabel("$t$", fontsize=16)
    ax.plot(ts, xs[:, 0], "-", label="$\\theta_1$")
    ax.plot(ts, xs[:, 1], "-", label="$\\theta_2$")
    ax.plot(ts, xs[:, 2], "-", label="$\dot \\theta_1$")
    ax.plot(ts, xs[:, 3], "-", label="$\dot \\theta_2$")
    ax.plot(ts[1:], us, "-", label="input")

    ax.legend(fontsize=16)
    ax.set_title(controller_name)
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

        
        
