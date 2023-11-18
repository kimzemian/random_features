"""Class factory for controlling an unknown system."""
from .eval import simulate_sys, eval_cs
from .train import train_episodic, train_episodic_with_info
from .data import *
from .init_controllers import *
from fast_control.util import load_config
from fast_control.gp_factory import GPFactory
from core.systems import DoubleInvertedPendulum


class ControllerFactory:
    """Control an unknown system using data driven methods."""

    def __init__(self, path):
        """Initialize an instance of the class."""
        self.config_path = path
        self._load_configuration(path)
        # Initialize controllers
        init_oracle_controller(self)
        init_qp_controller(self)
        # gp_factory = GPFactory()
        # self.gps_names = gp_factory.gp_param_dict.keys()

    def _load_configuration(self, path):
        """Load configuration from the TOML file."""
        gp_conf = load_config(path)["gps"]
        self.gps_names = gp_conf.keys()

        sys_conf = load_config(path)["controller_factory"]
        self.m = sys_conf["m"]
        self.sys_params = sys_conf["sys_params"]
        self.sys_est_params = sys_conf["sys_est_params"]
        self.system = DoubleInvertedPendulum(*sys_conf["sys_params"])
        self.system_est = DoubleInvertedPendulum(*sys_conf["sys_est_params"])

        episodic_conf = sys_conf["episodic"]
        self.epochs = episodic_conf["epochs"]
        self.episodic_T = episodic_conf["T"]
        self.episodic_num_steps = episodic_conf["num_steps"]
        self.x_0 = episodic_conf["x_0"]

        grid_conf = sys_conf["grid"]
        self.grid_T = grid_conf["T"]
        self.grid_num_steps = grid_conf["num_steps"]

        nominal_conf = sys_conf["nominal_controller"]
        self.nominal_static_cost = nominal_conf["static_cost"]
        self.nominal_regularizer = nominal_conf["regularizer"]
        self.nominal_coef = nominal_conf["coef"]
