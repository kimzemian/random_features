import sys

import matplotlib.pyplot as plt
import scienceplots

sys.path.append("..")


class ControlSystemPlotter:
    def __init__(self, x_0=None, gps_names=None):
        self.x_0 = x_0
        self.gps_names = gps_names
        plt.style.use("science")

    # def load_data(self, path, episodic=False):
    #     data = np.load(path)
    #     if episodic:
    #         gp_zs = data["gp_zs"][:, :, :, -1]
    #         qp_zs = data["qp_zs"][:, :, -1]
    #         model_zs = data["model_zs"][:, :, -1]
    #     else:
    #         gp_zs = data["gp_zs"][:, :, :]
    #         qp_zs = data["qp_zs"][:, :]
    #         model_zs = data["model_zs"][:, :]
    #     ts = data["ts"]
    #     return gp_zs, qp_zs, model_zs, ts

    def set_value(self, c_cdot):
        value = "$C(x)$" if c_cdot == 0 else "$\dot{C}$"
        val = "C" if c_cdot == 0 else "c_dot"
        return value, val

    def finalize_and_save(self, fig, ax, value, path, title=None):
        plt.ylabel(value, fontsize=8)
        plt.figtext(0.12, 0.94, f"x_0={self.x_0}")
        if not title:
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path)
        plt.show()
        plt.close()


from .plot import plot_info
