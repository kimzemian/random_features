from fast_control.viz.episodic_plot import *
from fast_control.viz.animate import create_animation
from fast_control.viz.plot import *
from fast_control.controller_factory import (
    eval_cs,
    ControllerFactory,
    create_grid_data,
    grid_info,
    train_grid,
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
system = DoubleInvertedPendulum(*sys_conf["sys_params"])
system_est = DoubleInvertedPendulum(*sys_conf["sys_est_params"])

control = ControllerFactory(system, system_est)
control.T = 1
control.num_steps = 10
create_grid_data(control)
control.T = sys_conf["T"]
control.num_steps = sys_conf["num_steps"]
controllers, gps = train_grid(control)
grid_info(control, controllers)
# with open("data/control.pickle", "wb") as handle:
#     pickle.dump(control, handle, protocol=pickle.HIGHEST_PROTOCOL)
path = "data/eval_cs_grid.npz"

plot_info(control.x_0, controllers, path, c_cdot=0)
plot_info(control.x_0, controllers, path, c_cdot=1)

serialized_controllers = dill.dumps(controllers)
serialized_control = dill.dumps(control)
with open("data/controllers.pickle", "wb") as handle:
    pickle.dump(serialized_controllers, handle)
with open("data/control.pickle", "wb") as handle:
    pickle.dump(serialized_control, handle)
# plot_predicted_vs_true_func(x_0, epochs, T)
# plot_qp(control, 0)
