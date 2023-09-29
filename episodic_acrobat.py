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
import toml
import pickle
import dill

np.set_printoptions(threshold=sys.maxsize)
with open("config.toml") as f:
    config = toml.load(f)
sys_conf = config["acrobat"]
gps_names = sys_conf["gps_names"]
system = DoubleInvertedPendulum(*sys_conf["sys_params"])
system_est = DoubleInvertedPendulum(*sys_conf["sys_est_params"])

control = ControllerFactory(system, system_est)
controllers, gps = train_episodic_with_info(control)

serialized_controllers = dill.dumps(controllers)
serialized_control = dill.dumps(control)
with open("data/control.pickle", "wb") as handle:
    pickle.dump(serialized_control, handle)
with open("data/controllers.pickle", "wb") as handle:
    pickle.dump(serialized_controllers, handle)

path = "data/eval_cs.npz"
plot_info(control.x_0, controllers, path, c_cdot=1)
plot_info(control.x_0, controllers, path, c_cdot=0)
episodic_plot_cum_predicted_vs_true_eval(
    control.x_0, control.epochs, control.T, control.num_steps
)
episodic_plot_cum_diff_from_oracle_for_controller(
    control.x_0, control.epochs, path, gps_names, c_cdot=0
)
episodic_plot_cum_diff_from_oracle_for_controller(
    control.x_0, control.epochs, path, gps_names, c_cdot=1
)
