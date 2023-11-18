"""episode plot methods."""
import pickle
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-whitegrid")


def episodic_plot_eval_for_controller(self, x_0, epochs, path, gps_names):
    """Plot true C_dot/true C for epochs."""
    cmap = plt.get_cmap("jet", epochs)
    norm = matplotlib.colors.BoundaryNorm(np.arange(epochs + 1) + 0.5, epochs)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    c = 1
    plot_path = "/home/kk983/fast_control/plots/"
    data = np.load(path)
    gp_zs = data["gp_zs"]
    # qp_zs = data["qp_zs"]
    model_zs = data["model_zs"][:, -1]
    ts = data["ts"]

    for gp_data, gp_name in zip(gp_zs, gps_names):
        fig, ax = plt.subplots(sharex=True)
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel("$C$", fontsize=8)

        for epoch in range(epochs):
            ax.plot(ts, gp_data[:, epoch], markersize=c, alpha=0.4, c=cmap(epoch))
        ax.plot(ts, model_zs, "k-.", label="oracle_controller", markersize=c)

        fig.colorbar(sm, ticks=np.linspace(0, epochs, 6))
        ax.legend()
        ax.set_title(f"True C/C_dot for {gp_name} controller over time, x_0={x_0}")
        # plt.figtext(0.12, 0.94, f"x_0={x_0},{gp_name}")
        fig.figsize = (9, 6)
        fig.tight_layout()
        fig.savefig(
            f"{plot_path}acrobat/{gp_name} controller over time -time{int(time.time())}.png"
        )
        plt.show()
        plt.close()


def episodic_plot_cum_diff_from_oracle_for_controller(
    self, x_0, epochs, path, gps_names, c_cdot=0
):
    """Plot true diff from oracle for true C_dot/C for epochs."""
    cmap = plt.get_cmap("jet", epochs)
    norm = matplotlib.colors.BoundaryNorm(np.arange(epochs + 1) + 0.5, epochs)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    c = 1
    plot_path = "/home/kk983/fast_control/plots/"
    data = np.load(path)  # FIXME
    gp_zs = data["gp_zs"]
    qp_zs = data["qp_zs"]
    model_zs = data["model_zs"]
    ts = data["ts"]
    diff_gp = gp_zs[:, :, :, :] - model_zs[:, np.newaxis, :, :]
    csum_gps = np.cumsum(diff_gp, axis=2)
    csum_qp = np.cumsum(qp_zs, axis=1)
    value = "$C$" if c_cdot == 0 else "$\dot{C}$"

    for i, gp_name in enumerate(gps_names):
        fig, ax = plt.subplots(sharex=True)
        ax.set_ylim(0, 3000) if c_cdot == 0 else ax.set_ylim(-1500, 2000)
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel(value, fontsize=8)

        for epoch in range(epochs):
            ax.plot(
                ts,
                csum_gps[c_cdot, i, :, epoch],
                markersize=c,
                alpha=0.4,
                c=cmap(epoch),
            )
        ax.plot(
            ts, model_zs[c_cdot, :, 0], "k-.", label="oracle_controller", markersize=c
        )
        ax.plot(ts, csum_qp[c_cdot, :, 0], "-", label="qp_controller", markersize=c)

        fig.colorbar(sm, ticks=np.linspace(0, epochs, 6))
        ax.legend()
        ax.set_title(
            f"cumdiff of {value} from oracle for {gp_name} controller over time, x_0={x_0}"
        )
        # plt.figtext(0.12, 0.94, f"x_0={x_0},{gp_name}")
        fig.figsize = (9, 6)
        fig.tight_layout()
        fig.savefig(
            f"{plot_path}acrobat/cumdiff_from_oracle_{gp_name}_time{int(time.time())}.png"
        )
        plt.show()
        plt.close()


def episodic_plot_predicted_vs_true_eval(self, x_0, epochs, T, num_steps):
    """Plot absolute value of true C_dot/true C minus the predicted C_dot over epochs."""
    with open("data/test_previous_gp.pickle", "rb") as handle:
        data = pickle.load(handle)
    value = next(iter(data.values()))
    epochs = len(value[0])
    ts = np.linspace(0, T, num_steps)
    ts = ts[1:]
    plot_path = "/home/kk983/fast_control/plots/"
    c = 1
    fmts = ["c-", "m-", "y-", "r-"]

    for key in data.keys():
        cmap = plt.get_cmap("jet", epochs)
        norm = matplotlib.colors.BoundaryNorm(np.arange(epochs + 1) + 0.5, epochs)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig, ax = plt.subplots(sharex=True)
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel("$C$", fontsize=8)
        for epoch in range(2, epochs):
            ax.plot(
                ts,
                data[key][:, epoch],
                markersize=c,
                alpha=0.4,
                c=cmap(epoch),
            )

        fig.colorbar(sm, ticks=np.linspace(0, epochs, 6))
        ax.legend()
        ax.set_title(
            f"abs error in predicted df for {key} controller over episodes, x_0={x_0}"
        )
        fig.figsize = (9, 6)
        fig.tight_layout()
        fig.savefig(
            f"{plot_path}acrobat/{key} predicted-true z-time{int(time.time())}.png"
        )
        plt.show()
        plt.close()


def episodic_plot_cum_predicted_vs_true_eval(self, x_0, epochs, T, num_steps, path):
    """Plot cumulative sum of absolute value of true C_dot/true C minus the predicted C_dot over epoch."""
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    value = next(iter(data.values()))
    epochs = len(value[0])
    ts = np.linspace(0, T, num_steps)
    ts = ts[1:]
    c = 1
    plot_path = "/home/kk983/fast_control/plots/"
    for key in data:
        cmap = plt.get_cmap("jet", epochs)
        norm = matplotlib.colors.BoundaryNorm(np.arange(epochs + 1) + 0.5, epochs)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig, ax = plt.subplots(sharex=True)
        # ax.set_ylim(0, 2000)
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel("$C$", fontsize=8)
        csum = np.cumsum(data[key], axis=0)
        for epoch in range(2, epochs):
            ax.plot(ts, csum[:, epoch], markersize=c, alpha=0.4, c=cmap(epoch))

        fig.colorbar(sm, ticks=np.linspace(0, epochs, 6))
        ax.legend()
        ax.set_title(
            f"cumulative abs error in predicted df for {key} controller over episodes, x_0={x_0}"
        )
        # plt.figtext(0.12, 0.94, f"x_0={x_0},{key}")
        fig.figsize = (9, 6)
        fig.tight_layout()
        fig.savefig(
            f"{plot_path}acrobat/{key} cum error in df pred-time{int(time.time())}.png"
        )
        plt.show()
        plt.close()


def episodic_plot_u_for_controller(self, epochs, path, gps_names):
    """Plot true C_dot/true C over epochs."""
    cmap = plt.get_cmap("jet", epochs)
    norm = matplotlib.colors.BoundaryNorm(np.arange(epochs + 1) + 0.5, epochs)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    c = 1
    plot_path = "/home/kk983/fast_control/plots/"
    data = np.load(path)
    us = data["us"]
    ts = data["ts"]

    for i, gp_name in enumerate(gps_names):
        fig, ax = plt.subplots(sharex=True)
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel("$C$", fontsize=8)

        for epoch in range(epochs):
            ax.plot(ts, us[:, i, epoch], markersize=c, alpha=0.4, c=cmap(epoch))
        # ax.plot(ts, us, "k-.", label="oracle_controller", markersize=c)

        fig.colorbar(sm, ticks=np.linspace(0, epochs, 6))
        ax.legend()
        ax.set_title(f"u for {gp_name} controller over time")
        # plt.figtext(0.12, 0.94, f"x_0={x_0},{gp_name}")
        fig.figsize = (9, 6)
        fig.tight_layout()
        fig.savefig(
            f"{plot_path}acrobat/u for {gp_name} controller over time -time{int(time.time())}.png"
        )
        plt.show()
        plt.close()
