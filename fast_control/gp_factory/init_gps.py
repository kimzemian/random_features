"""Initialize controllers."""
import sys

sys.path.append("/home/kk983/core")
import numpy as np
from fast_control.gp_factory import (
    ADKernel,
    ADRandomFeatures,
    ADPKernel,
    ADPRandomFeatures,
    VanillaKernel,
    VanillaRandomFeatures,
    ADPRFSketch,
    ADRFOne,
)


def init_gp(self, gp_name, datum, rf_d):
    """Initialize specified kernel."""
    # Define a dictionary to map gp_name to corresponding class
    gp_classes = {
        ADRandomFeatures.name: ADRandomFeatures,
        ADPRandomFeatures.name: ADPRandomFeatures,
        ADKernel.name: ADKernel,
        ADPKernel.name: ADPKernel,
        ADPRFSketch.name: ADPRFSketch,
        ADRFOne.name: ADRFOne,
    }

    # Check if gp_name is in the dictionary, and create an instance
    if gp_name in gp_classes:
        gp_param = self.gp_param_dict[gp_name]
        return gp_classes[gp_name](
            *datum, sgm=gp_param["sgm"], reg_param=gp_param["reg_param"], rf_d=rf_d
        )
    else:
        raise ValueError(f"Unsupported gp_name: {gp_name}")


def train_gps(self, data, rf_d=None):
    """Wrapper for init_gps."""
    if rf_d is None:
        _, _, zs = next(iter(data.values()))
        num = len(zs) // 9
        rf_d = num + 1 if num % 2 else num  # FIXME:rf_d
        rf_d = 2 * rf_d
        print(f"data size:{len(zs)}, rf_d is: {rf_d}")
    gps = []
    for gp_name, datum in data.items():
        gp = init_gp(self, gp_name, datum, rf_d)
        gp.train()
        gps.append(gp)
    return gps


def init_gp_dict(self, xs, ys, zs):
    """Initialize gp dictionary.
    return a dictionary of training data for each gp."""
    return {
        gp_name: (np.copy(xs), np.copy(ys), np.copy(zs))
        for gp_name in self.gp_param_dict
    }


# def init_trained_gps(gps_names, data, params, rf_d=None):
#     """Initialize gps."""
#     data = init_gp_dict(gps_names, *data)
#     gps = train_gps(data, params, rf_d)
#     return gps
