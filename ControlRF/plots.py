import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

from ControlRF.util import data_gen, simulate

def plot_simulation(system, controller, controller_name, x_0,
					T=20, num_steps=200):
	xs, us, ts = simulate(system, controller, x_0, T, num_steps)
	'''plotting simulated system trajectories using specified controller '''

	ax = plt.figure(figsize=(8, 6), tight_layout=True).add_subplot(1, 1, 1)
	ax.set_xlabel('$t$', fontsize=16)
	ax.plot(ts, xs[:,0], '-', label='theta')
	ax.plot(ts, xs[:,1], '-', label='theta dot')
	ax.plot(ts[1:], us, '-', label='input')
	
	ax.legend(fontsize=16)
	ax.set_title(controller_name)
	ax.grid()

	ax = plt.figure(figsize=(8, 6), tight_layout=True).add_subplot(1, 1, 1)
	ax.set_xlabel('$\\theta$', fontsize=16)
	ax.set_ylabel('$\dot \\theta$', fontsize=16)
	ax.plot(xs[:,0], xs[:,1], '-')
	ax.grid()

def plot_pred_errorbar(xs, ys, zs, gps):
	''' plotting gp prediction on training data'''
	
	c = 1
	fmts = ['r.','b.','g.','c.']

	plt.figure(figsize=(9, 9), dpi=240)
	for gp,fmt in zip(gps,fmts):
		plt.errorbar(zs, gp.test(xs, ys), gp.sigma(xs, ys), fmt=fmt, label=gp.__name__ + ' pred', alpha=.4, markersize=c, capsize=3)
	plt.plot(zs, zs, 'k.' , label = 'actual data', markersize =c+1)
	
	plt.xlabel('$\dot{C}$')
	plt.ylabel('$\hat{\dot{C}}$')
	
	plt.legend()
	plt.title(f'gp predictions for {len(xs)} data points')
	plt.draw()
	plt.savefig("pred-label",bbox_inches = "tight")
	plt.figure()

def plot_closed_loop_errorbar(system, aff_lyap, controller, gp, x_0, cut_off=0,
					T=20, num_steps=200):	
	'''plotting error bars for simulated system with specified controller'''
	c = 1
	all_xs, all_ys, all_zs = data_gen(system, controller, aff_lyap, x_0, T, num_steps)
	xs = all_xs[cut_off:]
	ys = all_ys[cut_off:]
	zs = all_zs[cut_off:]

	plt.figure(figsize=(8, 6))
	plt.errorbar(zs, gp.test(xs, ys), gp.sigma(xs, ys), fmt='r.', label=gp.__name__+' controller pred', alpha=.4, markersize=c, capsize=3)
	plt.plot(zs, zs, 'k.', label='actual data', markersize=c)

	plt.xlabel('$\dot{C}$')
	plt.ylabel('$\hat{\dot{C}}$')
	plt.legend()
	plt.draw()
	plt.savefig("pred-label-closed-loop",bbox_inches = "tight")
	plt.figure()

def plot_all_closed_loop_errorbars(system, aff_lyap, controllers, gps, x_0, cut_off=0,T=20, num_steps=200):
	'''plotting error bars for simulated system for all controllers'''
	c = 1
	fmts = ['c.','m.','y.','r.']
	zs_fmts = ['b.','g.','r.', 'k.']

	plt.figure(figsize=(8, 6))
	for controller,gp,fmt,z_fmt in zip(controllers,gps,fmts,zs_fmts):
		all_xs, all_ys, all_zs = data_gen(system, controller, aff_lyap, x_0, T, num_steps)
		xs = all_xs[cut_off:]
		ys = all_ys[cut_off:] 
		zs = all_zs[cut_off:]
		plt.errorbar(zs, gp.test(xs, ys), gp.sigma(xs, ys), fmt=fmt, label=gp.__name__+' controller pred', alpha=.4, markersize=c, capsize=3)
		plt.plot(zs, zs, z_fmt, label='actual data', markersize=c)

	plt.xlabel('$\dot{C}$')
	plt.ylabel('$\hat{\dot{C}}$')
	plt.legend()
	plt.draw()
	plt.savefig("pred-label-closed-loop",bbox_inches = "tight")
	plt.figure()
	
