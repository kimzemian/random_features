
import numpy as np

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