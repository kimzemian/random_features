import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


plt.style.use("seaborn-whitegrid")

    
def episodic_plot_func_for_controller(x_0, epochs, path, gp_names):
    """plotting true C_dot/true C using specified controller over time """
    
    cmap = plt.get_cmap("jet", epochs)
    norm = matplotlib.colors.BoundaryNorm(np.arange(epochs+1)+0.5,epochs)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)   
    c = 1

    data = np.load(path)
    gp_zs = data["gp_zs"]
    # qp_zs = data["qp_zs"]
    model_zs = data["model_zs"][:,-1]
    ts = data["ts"]
    
    for gp_data,gp_name in zip(gp_zs,gp_names):
        fig, ax = plt.subplots(sharex=True)
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel("$C$", fontsize=8)

        for epoch in range(epochs):
            ax.plot(ts, gp_data[:,epoch], markersize=c, alpha=0.4, c=cmap(epoch))
        ax.plot(ts, model_zs, "k-.", label="oracle_controller", markersize=c)

        fig.colorbar(sm, ticks=np.arange(1., epochs + 1))
        ax.legend()
        ax.set_title(f'True C/C_dot for {gp_name} controller over time, x_0={x_0}')
        # plt.figtext(0.12, 0.94, f"x_0={x_0},{gp_name}")
        fig.figsize = (9, 6)
        fig.tight_layout()
        fig.savefig(f"dip_plots/{gp_name} controller over time.png")
        plt.show()
        plt.close()
    
def episodic_plot_predicted_vs_true_func(x_0, epochs, T, num_steps):
    """plotting true C_dot/true C versus the predicted C_dot for each iteration"""
    
    with open('data/test_previous_gp.pickle', 'rb') as handle:
        data = pickle.load(handle)
    value = next(iter(data.values()))
    epochs = len(value[0])
    ts = np.linspace(0, T, num_steps)
    ts = ts[1:]
    
    c = 1
    fmts = ["c-", "m-", "y-", "r-"]

    
    for key in data.keys():
        cmap = plt.get_cmap("jet", epochs)
        norm = matplotlib.colors.BoundaryNorm(np.arange(epochs+1)+0.5,epochs)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig, ax = plt.subplots(sharex=True)
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel("$C$", fontsize=8)
        for epoch in range(2,epochs):
            ax.plot(ts, data[key][:,epoch], label=epoch, markersize=c, alpha=0.4, c=cmap(epoch))

        fig.colorbar(sm, ticks=np.arange(1, epochs + 1))
        ax.legend()
        ax.set_title(f'abs error in predicted df for {gp_name} controller over episodes, x_0={x_0}')
        plt.figtext(0.12, 0.94, f"x_0={x_0},{key}")
        fig.figsize = (9, 6)
        fig.tight_layout()
        fig.savefig(f"dip_plots/{key} predicted-true z.png")
        plt.show()
        plt.close() 
        

def episodic_plot_cum_predicted_vs_true_func(x_0, epochs, T, num_steps):
    """plotting true C_dot/true C versus the predicted C_dot for each iteration"""
    
    with open('data/test_previous_gp.pickle', 'rb') as handle:
        data = pickle.load(handle)
    value = next(iter(data.values()))
    epochs = len(value[0])
    ts = np.linspace(0, T, num_steps)
    ts = ts[1:]
    
    c = 1
    fmts = ["c-", "m-", "y-", "r-"]
    
    
    for key in data.keys():
        cmap = plt.get_cmap("jet", epochs)
        norm = matplotlib.colors.BoundaryNorm(np.arange(epochs+1)+0.5,epochs)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig, ax = plt.subplots(sharex=True)
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel("$C$", fontsize=8)
        csum = np.cumsum(data[key], axis=0)
        for epoch in range(2,epochs):
            ax.plot(ts, csum[:,epoch], label=epoch, markersize=c, alpha=0.4, c=cmap(epoch))

        fig.colorbar(sm, ticks=np.arange(1, epochs + 1))
        ax.legend()
        ax.set_title(f'cumulative abs error in predicted df for {key} controller over episodes, x_0={x_0}')
        # plt.figtext(0.12, 0.94, f"x_0={x_0},{key}")
        fig.figsize = (9, 6)
        fig.tight_layout()
        fig.savefig(f"dip_plots/{key} cum error in df pred.png")
        plt.show()
        plt.close()

        
        
