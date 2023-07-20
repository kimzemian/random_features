import re
import pickle
from tqdm import tqdm
from ControlRF import (
    GPController,
    ADPKernel,
    ADPRandomFeatures,
    ADKernel,
    ADRandomFeatures,
)
from ControlRF.data import *
from ControlRF.eval import *

def train(system_est, xs, ys, zs, sgm=10, slack='linear'):
    '''generates and trains data driven controllers given training data'''
    num = len(zs)//10
    m = len(ys[0]) - 1
    rf_d = num+1 if num%2 else num
    ad_rf = ADRandomFeatures(xs, ys, zs, sgm, m*rf_d)
    adp_rf = ADPRandomFeatures(xs, ys, zs, sgm, rf_d)
    ad_kernel = ADKernel(xs, ys, zs, sgm)
    adp_kernel = ADPKernel(xs, ys, zs, sgm)
    gps = [ad_kernel, ad_rf, adp_kernel, adp_rf]
    controllers = []
    gps_name = []
    for gp in gps:
        gps_name.append(gp.__name__)
        print(gp.__name__)
        gp_controller = GPController(system_est, gp)
        gp_controller.add_regularizer(system_est.fb_lin, 25)
        gp_controller.add_static_cost(np.identity(2))
        gp_controller.add_stability_constraint(
            system_est.lyap, comp=system_est.alpha, slack=slack, coeff=1e5
        )
        controllers.append(gp_controller)
        print(f"training time for {gp.__name__}_gp is: {gp.training_time}")
    return controllers, gps, gps_name


def train_episodic(system, system_est, x_0, epochs, T, num_steps,info=False, func=None, sgm=10, slack='linear'):
    '''episodically trains data driven controllers given initial training data, for specified epochs, info saves more information'''
    xs, ys, zs = training_data_gen(
        system, system_est, system.qp_controller, torch.from_numpy(x_0), T, num_steps
    )
    controllers, gps, gps_name = train(system_est, xs, ys, zs, sgm, slack)
    data = dict.fromkeys(gps_name)
    tested_data = dict.fromkeys(gps_name)
    for gp in gps_name:
        data[gp] = (xs,ys,zs)
        tested_data[gp] = np.empty((len(zs),epochs))
    if info:
        gp_zs = np.empty((4, num_steps, epochs))
        qp_zs = np.empty((num_steps, epochs)) 
        model_zs = np.empty((num_steps, epochs))
        diff_gp_zs = np.empty((4, epochs))
        diff_qp_zs = np.empty(epochs)
    

    for epoch in tqdm(range(epochs)):
        for controller,gp in zip(controllers,gps):
            x, y, z = training_data_gen(system, system_est, controller, xs[0], T, num_steps)
            
            xs, ys, zs = data[gp.__name__]
            xs = np.concatenate((xs, x))
            ys = np.concatenate((ys, y))
            zs = np.concatenate((zs, z))
            data[gp.__name__] = xs, ys, zs
            
            pred = gp.test(x,y)
            tested_data[gp.__name__][:,epoch] = np.abs(pred - z)
            
        controllers, gps, _ = train(system_est, xs, ys, zs, sgm, slack)
        print(f"iteration{epoch}:data size:{len(xs)}")
            
       #plot information for all epochs
        if info:   
            #compute func(true c_dot/true c) for controllers 
            gp_z, qp_z, model_z, ts = info_data_gen(xs[0], controllers, system, \
                                                    T, num_steps, func)
            gp_zs[:,:,epoch] = gp_z
            qp_zs[:,epoch] = qp_z
            model_zs[:,epoch] = model_z

            #computes sum of difference between the controllers and the oracle
            diff_gp_z, diff_qp_z = info_data_gen(xs[0], controllers, system, \
                            T, num_steps, func, info)
            diff_gp_zs[:,epoch] = diff_gp_z
            diff_qp_zs[epoch] = diff_qp_z
            
    #saves sum of difference between the controllers and the oracle
    if info:
        np.savez(f"data/{re.sub('.','',str(xs[0]))}_{func.__name__}", gp_zs=gp_zs, qp_zs=qp_zs, model_zs=model_zs, ts=ts) 
        np.savez(f"data/{re.sub('.','',str(xs[0]))}_diff from oracle", gp_zs=diff_gp_zs, qp_zs=diff_qp_zs, model_zs=np.zeros(epochs), ts=np.arange(epochs))
        with open('data/test_previous_gp.pickle', 'wb') as handle:
            pickle.dump(tested_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                  
    return controllers, gps, 







