import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import networkx as nx
from matplotlib import animation
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

plt.style.use("seaborn-whitegrid")

from ControlRF.util import *


def plot_simulation(system, controller, controller_name, x_0, T=20, num_steps=200):
    xs, us, ts = simulate(system, controller, x_0, T, num_steps)
    """plotting simulated system trajectories using specified controller """

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


def plot_info(num, x_0, controllers, path):
    """plotting true C_dot/C using specified controller over time"""

    c = 1
    fmts = ["c-", "m-", "y-", "r-"]

    fig, ax = plt.subplots(sharex=True)
    ax.set_xlabel("$t$", fontsize=8)
    ax.set_ylabel("$C$", fontsize=8)

    data = np.load(path)
    # data = np.load(f"data/[0,pi,0,0]/eval_c_{num}.npz")
    gp_zs = data["gp_zs"]
    qp_zs = data["qp_zs"]
    model_zs = data["model_zs"]
    ts = data["ts"]

    for i, (controller, fmt) in enumerate(zip(controllers, fmts)):
        ax.plot(ts, gp_zs[i], fmt, label=controller.__name__, markersize=c, alpha=0.4)

    ax.plot(ts, qp_zs, "-", label="qp_controller", markersize=c)
    ax.plot(ts, model_zs, "k-.", label="oracle_controller", markersize=c)

    ax.legend()
    plt.figtext(0.12, 0.94, f"x_0={x_0}")
    fig.figsize = (9, 6)
    fig.tight_layout()
    fig.savefig(f"data/[1,0,0,0]/{num}_10.png")
    plt.show()
    plt.close()

    fig, ax = plt.subplots(sharex=True)
    ax.set_xlabel("$t$", fontsize=8)
    ax.set_ylabel("$C$", fontsize=8)

    for i, (controller, fmt) in enumerate(zip(controllers, fmts)):
        ax.plot(
            ts[80:100],
            gp_zs[i, 80:100],
            fmt,
            label=controller.__name__,
            markersize=c,
            alpha=0.4,
        )
    ax.plot(ts[80:100], qp_zs[80:100], "-", label="qp_controller", markersize=c)
    ax.plot(ts[80:100], model_zs[80:100], "k-.", label="oracle_controller", markersize=c)

    ax.legend()
    plt.figtext(0.12, 0.94, f"x_0={x_0}")
    fig.figsize = (9, 6)
    fig.tight_layout()
    fig.savefig(f"data/[1,0,0,0]/{num}_3_5.png")
    plt.show()

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


def plot_c_dot_over_time(system, aff_lyap, controllers, gps, x_0, T=10, num_steps=200):
    """plotting true C_dot using specified controller over time"""
    c = 1
    ts = np.linspace(0, T, num_steps)[1:]
    fmts = ["c-", "m-", "y-", "r-"]
    zs_fmts = ["b.", "g.", "r.", "k."]

    ax = plt.figure(figsize=(8, 6), tight_layout=True).add_subplot(1, 1, 1)
    ax.set_xlabel("$t$", fontsize=16)

    for controller, gp, fmt in zip(controllers, gps, fmts):
        _, _, zs = data_gen(system, controller, aff_lyap, x_0, T, num_steps)
        ax.plot(ts, zs, fmt, label=gp.__name__ + " controller", markersize=c)

    ax.ylabel("true $\hat{\dot{C}}$")
    ax.legend()
    plt.show()
    # plt.savefig('all_closed_loop_errorbars')
    plt.close()


def render(system, controller, controller_name, x_0, T=20, num_steps=200):
    xs, us, ts = simulate(system, controller, x_0, T, num_steps)
    dt = T / num_steps

    x_solution = np.zeros(len(xs))
    a_solution = xs[:, 0]
    b_solution = xs[:, 2]

    skip_frames = 5

    x_solution = x_solution[::skip_frames]
    a_solution = a_solution[::skip_frames]
    b_solution = b_solution[::skip_frames]

    frames = len(x_solution)

    j1_x = l_1 * np.sin(a_solution) + x_solution
    j1_y = l_1 * np.cos(a_solution)

    j2_x = l_2 * np.sin(b_solution) + j1_x
    j2_y = l_2 * np.cos(b_solution) + j1_y

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1, 1), ylim=(-1, 1))
    ax.set_aspect("equal")
    ax.grid()

    patch = ax.add_patch(
        Rectangle((0, 0), 0, 0, linewidth=1, edgecolor="k", facecolor="r")
    )

    (line,) = ax.plot([], [], "o-", lw=2)
    time_template = "time: %.1f s"
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    cart_width = 0.15
    cart_height = 0.1

    def init():
        line.set_data([], [])
        time_text.set_text("")
        patch.set_xy((-cart_width / 2, -cart_height / 2))
        patch.set_width(cart_width)
        patch.set_height(cart_height)
        return line, time_text

        def animate(i):
            thisx = [x_solution[i], j1_x[i], j2_x[i]]
            thisy = [0, j1_y[i], j2_y[i]]

            line.set_data(thisx, thisy)
            now = i * skip_frames * dt
            time_text.set_text(time_template % now)

            patch.set_x(x_solution[i] - cart_width / 2)
            return line, time_text, patch

    ani = animation.FuncAnimation(
        fig, animate, frames=frames, interval=1, blit=True, init_func=init, repeat=False
    )
    plt.close(fig)
    return ani


def animate():
    ani = render(system, qp_controller, "qp_controller", x_0, T=100, num_steps=1000)
    # %time ani.save('dip_qp_2.gif', writer=animation.PillowWriter(fps=24))
    for gp, controller in tqdm(zip(gps, controllers)):
        ani = render(
            system, controller, f"{gp.__name__}_controller", x_0, T=100, num_steps=1000
        )
        # %time ani.save(f'dip_{gp.__name__}_2.gif', writer=animation.PillowWriter(fps=24))
