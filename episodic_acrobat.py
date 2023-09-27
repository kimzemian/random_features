from fast_control.viz.episodic_plot import *
from fast_control.viz.animate import create_animation
from fast_control.viz.plot import *
from fast_control.controller_factory import (
    eval_cs,
    ControllerFactory,
    train_episodic_with_info,
)
from fast_control.controller_factory import ControllerFactory
from core.systems import DoubleInvertedPendulum
import sys
import mosek
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

system = DoubleInvertedPendulum(0.6, 0.6, 0.6, 0.6)
system_est = DoubleInvertedPendulum(1, 1, 1, 1)

control = ControllerFactory(system, system_est)
controllers, gps = train_episodic_with_info(control)

path = "data/eval_cs.npz"
episodic_plot_predicted_vs_true_eval(
    control.x_0, control.epochs, control.T, control.num_steps
)
plot_info(control.x_0, controllers, path, c_cdot=1)
plot_info(control.x_0, controllers, "data/diff_from_oracle.npz", diff=True, c_cdot=1)
# plot_predicted_vs_true_func(x_0, epochs, T)
episodic_plot_cum_predicted_vs_true_eval(
    control.x_0, control.epochs, control.T, control.num_steps
)
plot_qp(control, 0)
