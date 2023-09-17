import sys

sys.path.append("/home/kk983/core")


from ControlRF.train import *
from ControlRF.viz.episodic_plot import *
from ControlRF.viz.animate import *
from ControlRF.viz.plot import *
from ControlRF.util import *
from ControlRF.eval import eval_c
from core.systems import InvertedPendulum
import sys
import mosek
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

system = InvertedPendulum(0.7, 0.7)
system_est = InvertedPendulum(0.50, 0.5)
system.lyap = AffineQuadCLF.build_care(system, Q=np.identity(2), R=np.identity(1))
system_est.lyap = AffineQuadCLF.build_care(
    system_est, Q=np.identity(2), R=np.identity(1)
)

system_est.alpha = 1 / max(la.eigvals(system_est.lyap.P))

# Nominal Controller Static Slacked
x_0 = np.array([2, 0.1])
system.qp_controller = QPController(system_est, system.m)
system.qp_controller.add_static_cost(np.identity(1))
system.qp_controller.add_stability_constraint(
    system_est.lyap, comp=lambda r: system_est.alpha * r, slacked=True, coeff=1e3
)
# plot_simulation(system, system_est.qp_controller, 'qp_controller', x_0, T=100, num_steps=1000)


system.oracle_controller = QPController(system, system.m)
system.oracle_controller.add_static_cost(np.identity(1))
system.oracle_controller.add_stability_constraint(
    system.lyap, comp=lambda r: system_est.alpha * r, slacked=True, coeff=1e3
)
# plot_simulation(system, system.oracle_controller, 'oracle_controller', x_0, T=100, num_steps=1000)

T = 20
num_steps = 100
epochs = 10

if __name__ == "__main__":
    gps_names = ["adp_kernel", "ad_kernel"]
    controllers, gps = train_episodic(
        system,
        system_est,
        x_0,
        epochs,
        T,
        num_steps,
        gps_names,
        info=True,
        func=eval_c,
        sgm=1,
        slack="constant",
        D=1,
        coeff=1e3,
    )
    path = "data/eval_c.npz"
    plot_info(x_0, controllers, path)
    plot_info(x_0, controllers, "data/diff_from_oracle.npz", diff=True)
    # plot_predicted_vs_true_func(x_0, epochs, T)
    episodic_plot_cum_predicted_vs_true_func(x_0, epochs, T, num_steps)
    episodic_plot_predicted_vs_true_func(x_0, epochs, T, num_steps)
