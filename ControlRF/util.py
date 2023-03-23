
import numpy as np
import torch

def simulate(system, controller, x_0, T, num_steps):
    '''simulate system with specified controller'''

    ts = np.linspace(0, T, num_steps)
    xs, us = system.simulate(x_0, controller, ts)
    return xs, us, ts

def build_ccf_data(aff_lyap, xs, us, ts):
    ''' estimate error in the derivate of the CCF function
    using forward differencing '''

    av_x = (xs[:-1] + xs[1:])/2
    zs = [(aff_lyap.eval(xs[i+1],ts[i+1])- aff_lyap.eval(xs[i],ts[i]))/(ts[i+1]-ts[i]) - \
    aff_lyap.eval_dot(av_x[i],us[i],ts[i]) for i in range(len(us))]
    ys = np.concatenate((np.ones((len(us),1)), us), axis=1)
    return av_x, ys, zs

def data_gen(system, controller, aff_lyap, x_0, T, num_steps):
    '''compounding build_ccf_data and simulate methods to 
    generate training data for the gps '''

    xs, us, ts = simulate(system, controller, x_0, T, num_steps)
    xs, ys, zs = build_ccf_data(aff_lyap, xs, us, ts) #ts=ts[1:-1]
    return xs, ys, zs

def create_data(system, qp_controller, lyap_est, T, num_steps):  
    initial_x0s = np.mgrid[.1:np.pi:1, -1:1.1:.4, 0:np.pi:1, -1:1.1:.4].reshape(4, -1).T
    xs, ys, zs = data_gen(system, qp_controller, lyap_est, torch.from_numpy(initial_x0s[0]),  T, num_steps)
    for x_0 in initial_x0s[1:]:
        x, y, z = data_gen(system, qp_controller, lyap_est, torch.from_numpy(x_0),  T, num_steps)
        xs = np.concatenate((xs,x))
        ys = np.concatenate((ys,y))
        zs = np.concatenate((zs,z))

    initial_x0s = np.mgrid[-.1:.15:.05, -.1:.15:.05, -.1:.15:.05, -.1:.15:.05].reshape(4, -1).T
    for x_0 in initial_x0s:
        x, y, z = data_gen(system, qp_controller, lyap_est, torch.FloatTensor([0.1,0,0,0]), 10, 100)
        xs = np.concatenate((xs,x))
        ys = np.concatenate((ys,y))
        zs = np.concatenate((zs,z))

    np.savez(f'data_{T}_{num_steps}', xs=xs, ys=ys, zs=zs)


def c_dot(system, controller, aff_lyap, x_0, T, num_steps):
    '''returns true C_dot for simulated data with specified controller'''
    xs, us, ts = simulate(system, controller, x_0, T, num_steps)
    av_x = (xs[:-1] + xs[1:])/2
    zs = [(aff_lyap.eval(xs[i+1],ts[i+1])- aff_lyap.eval(xs[i],ts[i]))/(ts[i+1]-ts[i]) for i in range(len(us))]
    return zs
