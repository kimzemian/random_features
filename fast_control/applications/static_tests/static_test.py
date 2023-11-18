"""
This module loads adp_kernel data, splits it into test and train sets, 
train rf methods on the train set and plots the mse/mae.
"""

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from fast_control.gp_factory import (
    ADKernel,
    ADPKernel,
    ADPRandomFeatures,
    ADRandomFeatures,
    ADPRFSketch,
    ADRFOne,
)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_and_split_data(data_path):
    # load adp_kernel data
    # with open(data_path + "50/raw_episodic_data.pkl", "rb") as f:
    #     data = pickle.load(f)
    # xs, ys, zs = data["adp_kernel"]
    grid_data = np.load(data_path + "init_grid_medium.npz")
    xs, ys, zs = grid_data["xs"], grid_data["ys"], grid_data["zs"]
    # test/train split
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        np.asarray(xs), np.asarray(ys), np.asarray(zs), test_size=0.25, shuffle=True
    )
    print(zs.shape)
    return x_train, x_test, y_train, y_test, z_train, z_test
    
    # print(z_test.shape)  # 2538
    # print(z_train.shape) #7614


# init_gp_dict, init_gp, train and test gp
def init_gp_dict(xs, ys, zs, gps_names):
    """Initialize gp dictionary."""
    data = dict.fromkeys(gps_names)
    for gp_name in gps_names:
        data[gp_name] = (np.copy(xs), np.copy(ys), np.copy(zs))
    return data


def init_gp(gp_name, datum, rf_d=None):
    """Initialize specified kernel."""
    if gp_name == ADRandomFeatures.name:
        return ADRandomFeatures(*datum, rf_d=rf_d)
    elif gp_name == ADPRandomFeatures.name:
        return ADPRandomFeatures(*datum, rf_d=rf_d)
    elif gp_name == ADKernel.name:
        return ADKernel(*datum)
    elif gp_name == ADPKernel.name:
        return ADPKernel(*datum)
    elif gp_name == ADPRFSketch.name:
        return ADPRFSketch(*datum, rf_d=rf_d)
    elif gp_name == ADRFOne.name:
        return ADRFOne(*datum, rf_d=rf_d)


def train_gps(data, rf_d=None):
    """Wrapper for init_gps."""
    gps = []
    for gp_name, datum in data.items():
        gp = init_gp(gp_name, datum, rf_d)
        gp.train()
        gps.append(gp)
    return gps


def plot(data, name):
    plot_path = "/home/kk983/fast_control/plots/"
    steps = data.shape[1]
    xs = np.arange(100, steps*101, 100)
    xs = np.tile(xs, 10)
    plt.figure()
    sns.lineplot(
        x=xs,
        y=data[0, :, :].flatten("F"),
        estimator=np.median,
        errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
        label="adp_rf",
    )
    sns.lineplot(
        x=xs,
        y=data[1, :, :].flatten("F"),
        estimator=np.median,
        errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
        label="ad_rf",
    )
    sns.lineplot(x=xs, y=data[2, :, :].flatten("F"), label="adp_kernel")
    sns.lineplot(x=xs, y=data[3, :, :].flatten("F"), label="ad_kernel")
    plt.xlabel("number of random features")
    plt.ylabel(f"{name}")
    plt.tight_layout()
    plt.legend()
    plt.savefig(plot_path + f"static_tests/{name}_grid.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    data_path = "/home/kk983/fast_control/data/"
    x_train, x_test, y_train, y_test, z_train, z_test = load_and_split_data(data_path+"grid/")
    num_steps = len(z_train) // 100
    rmse = np.empty([4, num_steps, 10])
    mae = np.empty([4, num_steps, 10])

    # train rf_gps
    gps_names = ["adp_rf", "ad_rf"]
    data = init_gp_dict(x_train, y_train, z_train, gps_names)

    for num in tqdm(range(num_steps)):
        for seed in range(10):
            gps = train_gps(data, 100 * (num + 1))  # get seed
            for i, gp in enumerate(gps):
                z_pred = gp.test(x_test, y_test)
                rmse[i, num, seed] = mean_squared_error(z_test, z_pred, squared=False)
                mae[i, num, seed] = mean_absolute_error(z_test, z_pred)

    # benchmark with trained adp_kernel
    data = init_gp_dict(x_train, y_train, z_train, ["adp_kernel", "ad_kernel"])
    gps = train_gps(data)
    for i, gp in enumerate(gps):
        z_pred = gp.test(x_test, y_test)
        rmse[i + 2, :, :] = mean_squared_error(z_test, z_pred, squared=False)
        mae[i + 2, :, :] = mean_absolute_error(z_test, z_pred)

    np.save(data_path + "static_tests/rmse_grid_10_tests.npy", rmse)
    np.save(data_path + "static_tests/mae_grid_10_tests.npy", mae)
    # rmse = np.load(data_path + "static_tests/rmse_grid_10_tests.npy")
    # mae = np.load(data_path + "static_tests/mae_grid_10_tests.npy")
    plot(rmse, "rmse")
    plot(mae, "mae")
