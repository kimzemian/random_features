"""Initialize controllers."""
import sys

sys.path.append("/home/kk983/core")
import numpy as np
import numpy.linalg as la
from core.controllers import FBLinController, LQRController, QPController
from core.dynamics import AffineQuadCLF
from fast_control import GPController
from fast_control.gps import (
    ADKernel,
    ADPKernel,
    ADPRandomFeatures,
    ADRandomFeatures,
    ADPRFSketch,
)


def init_oracle_controller(self):
    """Initialize oracle controller."""
    if self.m == 1:
        self.system.lyap = AffineQuadCLF.build_care(
            self.system, Q=np.identity(2), R=np.identity(1)
        )
        self.system.alpha = 1 / max(la.eigvals(self.system_est.lyap.P))
        self.system.oracle_controller = QPController(self.system, self.system.m)
        self.system.oracle_controller.add_static_cost(np.identity(1))
        self.system.oracle_controller.add_stability_constraint(
            self.system.lyap,
            comp=lambda r: self.system.alpha * r,
            slacked=True,
            coeff=1e3,
        )
    elif self.m == 2:
        Q, R = 10 * np.identity(4), np.identity(2)
        self.system.lyap = AffineQuadCLF.build_care(self.system, Q, R)
        self.system.alpha = min(la.eigvalsh(Q)) / max(la.eigvalsh(self.system.lyap.P))
        lqr = LQRController.build(self.system, Q, R)
        self.system.fb_lin = FBLinController(self.system, lqr)
        self.system.oracle_controller = QPController.build_care(self.system, Q, R)
        self.system.oracle_controller.add_regularizer(self.system.fb_lin, 25)
        # self.system.oracle_controller.add_static_cost(100* np.identity(2))
        self.system.oracle_controller.add_stability_constraint(
            self.system.lyap,
            comp=lambda r: self.system.alpha * r,
            slacked=True,
            coeff=1e6,
        )
    self.system.oracle_controller.name = "oracle_controller"


def init_qp_controller(self):
    """Initialize QP controller."""
    if self.m == 1:
        self.system_est.lyap = AffineQuadCLF.build_care(
            self.system_est, Q=np.identity(2), R=np.identity(1)
        )
        self.system_est.alpha = 1 / max(la.eigvals(self.system_est.lyap.P))
        self.system.qp_controller = QPController(self.system_est, self.system.m)
        self.system.qp_controller.add_static_cost(np.identity(1))
        self.system.qp_controller.add_stability_constraint(
            self.system_est.lyap,
            comp=lambda r: self.system_est.alpha * r,
            slacked=True,
            coeff=1e3,
        )

    elif self.m == 2:
        Q, R = 10 * np.identity(4), np.identity(2)
        self.system_est.lyap = AffineQuadCLF.build_care(self.system_est, Q, R)
        self.system_est.alpha = min(la.eigvalsh(Q)) / max(
            la.eigvalsh(self.system_est.lyap.P)
        )
        model_lqr = LQRController.build(self.system_est, Q, R)
        self.system_est.fb_lin = FBLinController(self.system_est, model_lqr)
        self.system.qp_controller = QPController.build_care(self.system_est, Q, R)
        self.system.qp_controller.add_regularizer(self.system_est.fb_lin, 25)
        self.system.qp_controller.add_static_cost(np.identity(2))
        self.system.qp_controller.add_stability_constraint(
            self.system_est.lyap,
            comp=lambda r: self.system_est.alpha * r,
            slacked=True,
            coeff=1e6,
        )
    self.system.qp_controller.name = "qp_controller"


def init_gp(self, gp_name, datum, rf_d):
    """Initialize specified kernel."""
    if gp_name == ADRandomFeatures.name:
        return ADRandomFeatures(*datum, self.m * rf_d)
    elif gp_name == ADPRandomFeatures.name:
        return ADPRandomFeatures(*datum, rf_d)
    elif gp_name == ADKernel.name:
        return ADKernel(*datum)
    elif gp_name == ADPKernel.name:
        return ADPKernel(*datum)
    elif gp_name == ADPRFSketch.name:
        return ADPRFSketch(*datum, rf_d)
    # use match if python version allows


def init_gpcontroller(self, gp):
    """Initialize controlller for a given gp."""
    print(gp.name)
    gp_controller = GPController(self.system_est, gp)
    if self.m == 2:
        gp_controller.add_regularizer(self.system_est.fb_lin, 25)
        gp_controller.add_static_cost(np.identity(2))
    elif self.m == 1:
        gp_controller.add_static_cost(np.identity(1))

    gp_controller.add_stability_constraint()
    print(f"training time for {gp.name}_gp is: {gp.training_time}")
    return gp_controller


def init_gpcontroller_pairs(self, data):
    """Wrap init_gp and init_gpcontroller."""
    _, _, zs = next(iter(data.values()))
    num = len(zs) // 10
    rf_d = num + 1 if num % 2 else num  # TODO:make rf_d user determinable
    print(f"data size:{len(zs)}, rf_d is: {rf_d}")
    gps = []
    controllers = []
    for gp_name, datum in data.items():
        gp = init_gp(self, gp_name, datum, rf_d)
        controller = init_gpcontroller(self, gp)
        gps.append(gp)
        controllers.append(controller)
    return controllers, gps


def init_gp_dict(self, xs, ys, zs):
    """Initialize gp dictionary."""
    data = dict.fromkeys(self.gps_names)
    for gp_name in self.gps_names:
        data[gp_name] = (np.copy(xs), np.copy(ys), np.copy(zs))
    return data
