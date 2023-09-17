import numpy as np
import numpy.linalg as la
from core.controllers import QPController, LQRController, FBLinController
from core.dynamics import AffineQuadCLF


def init_controllers(system, system_est):
    """initializes system.lyap, system_est.lyap
    system.alpha, system_est.alpha
    system.qp_controller, system.oracle_controller
    for two dimensional states"""
    Q, R = 10 * np.identity(4), np.identity(2)
    system_est.lyap = AffineQuadCLF.build_care(system_est, Q, R)
    system_est.alpha = min(la.eigvalsh(Q)) / max(la.eigvalsh(system_est.lyap.P))

    model_lqr = LQRController.build(system_est, Q, R)
    system_est.fb_lin = FBLinController(system_est, model_lqr)

    system.qp_controller = QPController.build_care(system_est, Q, R)
    system.qp_controller.add_regularizer(system_est.fb_lin, 16)
    system.qp_controller.add_static_cost(1e2 * np.identity(2))
    system.qp_controller.add_stability_constraint(
        system_est.lyap, comp=lambda r: system_est.alpha * r, slacked=True, coeff=1e6
    )

    system.lyap = AffineQuadCLF.build_care(system, Q, R)
    system.alpha = min(la.eigvalsh(Q)) / max(la.eigvalsh(system.lyap.P))
    lqr = LQRController.build(system, Q, R)
    system.fb_lin = FBLinController(system, lqr)
    system.oracle_controller = QPController.build_care(system, Q, R)
    system.oracle_controller.add_regularizer(system.fb_lin, 16)
    system.oracle_controller.add_static_cost(1e2 * np.identity(2))
    system.oracle_controller.add_stability_constraint(
        system.lyap, comp=lambda r: system.alpha * r, slacked=True, coeff=1e6
    )

    return None
