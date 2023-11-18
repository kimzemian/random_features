from fast_control.viz.episodic_plot import *
from fast_control.viz.animate import create_animation
from fast_control.viz.plot import *
from fast_control.controller_factory import (
    ControllerFactory,
    simulate_sys
)
from fast_control.gp_factory import init_gp_dict, train_gps, GPFactory
from core.systems import DoubleInvertedPendulum
import sys
import mosek
import numpy as np
import toml
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error


def plot(preds, labels, ts, gps_names):
    plot_path ="/home/kk983/fast_control/plots/"
    for i in range(4):
        plt.figure()
        for gp_num,name in enumerate(gps_names):
            sns.lineplot(x=ts,y=preds[gp_num,:,i],label=name+" $\dot{x}$ prediction")
        sns.lineplot(x=ts,y=labels[:,i],color='black', linestyle='--',label="true $\dot{x}$")
   
        plt.xlabel("time")
        plt.ylabel(f"{i}-th coordinate of x_dot")
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{plot_path}learn_sys/x_dot_error_{i}.png", dpi=300)
        plt.show()
        plt.close()


if __name__ == "__main__":
    with open("config.toml") as f:
        config = toml.load(f)
    sys_conf = config["controller_factory"]
    gps_names = config['gps'].keys()
    system = DoubleInvertedPendulum(*sys_conf["sys_params"])
    system_est = DoubleInvertedPendulum(*sys_conf["sys_est_params"])
    T ,num_steps = sys_conf["T"], sys_conf["num_steps"]
    ts = np.linspace(0, T, num_steps)
    control = ControllerFactory(system, system_est)
    # xs, ys, x_dots = create_grid_data(control, T, num_steps)
    grid_data = np.load("/home/kk983/fast_control/data/learn_sys/init_grid_medium.npz")
    xs, ys, x_dots = grid_data['xs'], grid_data['ys'], grid_data['x_dots']
    gp_factory = GPFactory()
    data = init_gp_dict(gp_factory, xs, ys, x_dots)
    gps = train_gps(gp_factory, data)
    #test on a random point
    x_test = [.5, .5, .5, .5]
    xs, ys, x_dots = training_data_gen(
            control, control.system.qp_controller, torch.FloatTensor(x_test)
        )
    
    #compute rmse and mae for each gp for each dimension of the labels on each point
    #plot the error for each dimension of the labels
    x_dot_preds = np.empty((len(gps), len(x_dots), 4))
    for i,gp in enumerate(gps):
        x_dot_preds[i,:,:] = gp.test(xs, ys)
        rmse = mean_squared_error(x_dots, x_dot_preds[i,:,:], squared=False, multioutput='raw_values')
        mae = mean_absolute_error(x_dots, x_dot_preds[i,:,:], multioutput='raw_values')
        print(gp.name, f"rmse is {rmse}")
        print(gp.name, f"mae is {mae}")
    
    plot(x_dot_preds, x_dots, ts[:-1], gps_names)
