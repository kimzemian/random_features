import numpy as np
import numpy.linalg as la
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from core.controllers import QPController, LQRController, FBLinController
from core.dynamics import AffineQuadCLF
from core.systems import DoubleInvertedPendulum


if __name__ == "__main__":
    system_est = DoubleInvertedPendulum(0.5, 0.5, 0.8, 0.8)
    Q, R = 10 * np.identity(4), np.identity(2)
    lyap_est = AffineQuadCLF.build_care(system_est, Q, R)
    alpha = min(la.eigvalsh(Q)) / max(la.eigvalsh(lyap_est.P))

    model_lqr = LQRController.build(system_est, Q, R)
    model_fb_lin = FBLinController(system_est, model_lqr)
    # fb_lin_data = system.simulate(x_0, fb_lin, ts) #1

    qp_controller = QPController.build_care(system_est, Q, R)
    qp_controller.add_regularizer(model_fb_lin, 25)
    qp_controller.add_static_cost(np.identity(2))
    qp_controller.add_stability_constraint(
        lyap_est, comp=lambda r: alpha * r, slacked=True, coeff=1e5
    )
    # plot_simulation_dip(system, controller, 'qp_controller', x_0, T=100, num_steps=1000)s

    l_1 = 1
    l_2 = 1
    system = DoubleInvertedPendulum(0.2, 0.2, l_1, l_2)
    lyap = AffineQuadCLF.build_care(system, Q, R)
    alpha = min(la.eigvalsh(Q)) / max(la.eigvalsh(lyap.P))

    lqr = LQRController.build(system, Q, R)
    fb_lin = FBLinController(system, lqr)
    oracle_controller = QPController.build_care(system, Q, R)
    oracle_controller.add_regularizer(fb_lin, 25)
    oracle_controller.add_static_cost(np.identity(2))
    oracle_controller.add_stability_constraint(
        lyap, comp=lambda r: alpha * r, slacked=True, coeff=1e5
    )

    x_0 = [0, np.pi, 0 , 0]
    T = 10
    num_steps = 100
    ts = np.linspace(0, T, num_steps)


    xs, ys, zs = train_data_gen(
            system, qp_controller, lyap_est, torch.from_numpy(x_0), T, num_steps
        )
    train_episodic(system_est, lyap_est, aff_lyap, fb_lin, alpha, xs, ys, zs, epochs, T, num_steps)


    

    # # initial_x0s = np.array([[1,0,0,0],[1,0,1,0],[0,0,1,0]])
    # make_c_data(3, np.array([0, -1, 2.8, -0.5]))
    # make_c_dot_data(3, np.array([0, -1, 2.8, -0.5]))
    # make_c_data(0, np.array([1, 0, 0, 0]))
    # make_c_dot_data(0, np.array([1, 0, 0, 0]))
    # # make_c_data(4,np.array([1,0,2,0]))
    # # make_c_dot_data(4,np.array([1,0,2,0]))
    # plot_c_dot("c_3", np.array([0, -1, 2.8, -0.5]))
    # # plot_c_dot('c_dots_3',np.array([0,-1,2.8,-.5]))
    # plot_c_dot("c_0", np.array([1, 0, 0, 0]))
    # # plot_c_dot('c_dots_0',np.array([1,0,0,0]))
    # # plot_c_dot('c_4',np.array([1,0,2,0]))
    # # plot_c_dot('c_dots_4',np.array([1,0,2,0]))

    num = 1
    info_data_gen(num, x_0, ts, controllers, system, lyap_est, T, num_steps, qp_controller, oracle_controller, lyap, eval_c)
    plot_info(num,x_0)

