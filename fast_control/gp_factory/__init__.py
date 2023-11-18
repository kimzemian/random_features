from fast_control.util import load_config


class GPFactory:
    """Factory for GP kernels."""

    def __init__(self):
        sys_conf = load_config()
        self.gp_param_dict = sys_conf["gps"]


from .ad_kernel import ADKernel
from .ad_rf import ADRandomFeatures
from .adp_kernel import ADPKernel
from .adp_rf import ADPRandomFeatures
from .vanilla_kernel import VanillaKernel
from .vanilla_rf import VanillaRandomFeatures
from .adp_rf_sketch import ADPRFSketch
from .ad_rf_one_kernel import ADRFOne
from .init_gps import init_gp_dict, train_gps
