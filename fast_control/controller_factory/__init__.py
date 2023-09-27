"""Class factory for controlling an unknown system."""
from .eval import simulate_sys, eval_cs, eval_all
from .train import train_episodic, train_episodic_with_info, train_grid
from .data import *
from .init_controllers import *
import toml


class ControllerFactory:
    """Control an unknown system using data driven methods."""

    def __init__(self, system, system_est):
        """Initialize an instance of class."""
        self.system = system
        self.system_est = system_est
        # Load the configuration from the TOML file
        with open("config.toml") as f:
            config = toml.load(f)
        sys_conf = (
            config["inverted_pendulum"] if system_est.m == 1 else config["acrobat"]
        )
        self.T = sys_conf["T"]
        self.num_steps = sys_conf["num_steps"]
        self.x_0 = sys_conf["x_0"]
        self.m = sys_conf["m"]
        self.gps_names = sys_conf["gps_names"]
        self.epochs = sys_conf["epochs"]
        init_oracle_controller(self)
        init_qp_controller(self)
