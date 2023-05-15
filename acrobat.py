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

    # create_grid_data(system, qp_controller, lyap_est, 1, 10)
    # data = np.load('data,pi,1,sim:10,100.npz')
    data = np.load("data/training_data/data_1_10,(10809, 4).npz")
    xs = data["xs"]
    ys = data["ys"]
    zs = data["zs"]

    xs_train, xs_test, ys_train, ys_test, zs_train, zs_test = train_test_split(
        xs, ys, zs, test_size=0.1, random_state=2
    )

    controllers, gps = train(system_est, lyap_est, xs_train, ys_train, zs_train)
    # with open('controllers_gps','wb') as f:
    #     pickle.dump(gps,f)

    for gp in gps:
        zs_pred = gp.test(xs_test, ys_test)
        diff = np.abs(zs_pred - zs_test)
        print(np.sum(np.abs(diff)) / len(zs_test))
        print(np.sum(diff < 1))
        # print(i,diff[i],zs_test[i], zs_pred[i])
        mse = mean_squared_error(zs_test, zs_pred)
        rmse = mean_squared_error(zs_test, zs_pred, squared=False)
        print(f"{gp.__name__} mse,rmse is {mse,rmse} respectively")

    initial_x0s = np.mgrid[-1:1.1:1, -1:1.1:0.5, -1:1.1:1, -1:1.1:0.5].reshape(4, -1).T
    T = 10
    num_steps = 100
    ts = np.linspace(0, T, num_steps)

    # initial_x0s = np.array([[1,0,0,0],[1,0,1,0],[0,0,1,0]])
    make_c_data(3, np.array([0, -1, 2.8, -0.5]))
    make_c_dot_data(3, np.array([0, -1, 2.8, -0.5]))
    make_c_data(0, np.array([1, 0, 0, 0]))
    make_c_dot_data(0, np.array([1, 0, 0, 0]))
    # make_c_data(4,np.array([1,0,2,0]))
    # make_c_dot_data(4,np.array([1,0,2,0]))
    plot_c_dot("c_3", np.array([0, -1, 2.8, -0.5]))
    # plot_c_dot('c_dots_3',np.array([0,-1,2.8,-.5]))
    plot_c_dot("c_0", np.array([1, 0, 0, 0]))
    # plot_c_dot('c_dots_0',np.array([1,0,0,0]))
    # plot_c_dot('c_4',np.array([1,0,2,0]))
    # plot_c_dot('c_dots_4',np.array([1,0,2,0]))
