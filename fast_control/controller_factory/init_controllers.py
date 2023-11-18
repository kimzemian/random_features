"""Initialize controllers."""
import sys

sys.path.append("/home/kk983/core")
import numpy as np
import numpy.linalg as la
from core.controllers import FBLinController, LQRController, QPController

# from core.systems import DoubleInvertedPendulum, InvertedPendulum
from core.dynamics import AffineQuadCLF

from fast_control.gp_factory import train_gps

from .gp_controller import GPController

# def init_systems(self):
#     if self.m == 1:
#         self.system = DoubleInvertedPendulum(*self.sys_params)
#         self.system_est = DoubleInvertedPendulum(*self.sys_params)
#     else:
#         self.system = InvertedPendulum(*self.sys_params)
#         self.system_est = InvertedPendulum(*self.sys_params)


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


def init_qp_controller(self):  # FIXME update hard coded params with self.nominal params
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
        self.system.qp_controller.add_regularizer(
            self.system_est.fb_lin, self.nominal_regularizer
        )
        self.system.qp_controller.add_static_cost(
            self.nominal_static_cost * np.identity(2)
        )
        self.system.qp_controller.add_stability_constraint(
            self.system_est.lyap,
            comp=lambda r: self.system_est.alpha * r,
            slacked=True,
            coeff=self.nominal_coef,
        )
    self.system.qp_controller.name = "qp_controller"


def init_gpcontroller(self, gp):
    """Initialize controlller for a given gp."""
    print(gp.name)
    gp_controller = GPController(self.system_est, gp, self.config_path)
    if self.m == 2:
        gp_controller.add_regularizer(self.system_est.fb_lin, 25)
        gp_controller.add_static_cost(np.identity(2))
    elif self.m == 1:
        gp_controller.add_static_cost(np.identity(1))

    gp_controller.add_stability_constraint()
    print(f"training time for {gp.name}_gp is: {gp.training_time}")
    return gp_controller


def train_controllers(self, gp_factory, data, rf_d=None):
    """Wrapper for init_gpcontroller."""
    gps = train_gps(gp_factory, data, rf_d)
    controllers = []
    for gp in gps:
        controller = init_gpcontroller(self, gp)
        controllers.append(controller)
    return controllers, gps
